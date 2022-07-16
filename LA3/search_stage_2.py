import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

import logging
import math
import os

from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel

from theconf import Config as C, ConfigArgumentParser

from LA3.common import get_logger

from LA3.augmentations import *
import random
import pickle as pickle
from sklearn.metrics import pairwise_distances


logger = get_logger('LA3')
logger.setLevel(logging.INFO)


def train_neural_predictors_batch(ensemble_np_lst, ensemble_np_opt_lst, total_epoch, aug_pg_data):

    class CustomLoader:
        def __init__(self, data, batch_size):
            self.data = data
            self.batch_size = batch_size
            self.num_batches = int(math.ceil(len(self.data) / batch_size))
            random.shuffle(data)

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            for i in range(self.num_batches):
                batch = self.data[i*self.batch_size: (i+1)*self.batch_size]
                x, y = zip(*batch)
                yield x, y

    batch_size = 10000
    data = list(zip([d[:5] for d in aug_pg_data], [d[-1:] for d in aug_pg_data]))
    random.shuffle(data)
    train_num = int(len(data) * 0.9)
    train_loader = CustomLoader(data[:train_num], batch_size)
    valid_loader = CustomLoader(data[train_num:], batch_size)

    loss_fn = nn.L1Loss()

    for e_np_i in range(len(ensemble_np_lst)):
        print("e_np_i: ", e_np_i)
        for pg_epoch in range(total_epoch):
            ensemble_np_lst[e_np_i].train()

            pg_epo_loss = 0.0

            for x, y in train_loader:
                aug_encoding = torch.Tensor(x).long()
                pg_val = torch.Tensor(y).float()
                if torch.cuda.is_available():
                    aug_encoding, pg_val = aug_encoding.cuda(), pg_val.cuda()

                pg_preds = ensemble_np_lst[e_np_i](aug_encoding)

                pg_loss = loss_fn(pg_preds, pg_val)

                pg_loss.backward()
                ensemble_np_opt_lst[e_np_i].step()
                ensemble_np_opt_lst[e_np_i].zero_grad()

                pg_epo_loss += pg_loss.item() * aug_encoding.shape[0]

                del aug_encoding, pg_val, pg_preds

            if pg_epoch % 10 == 0 or pg_epoch == total_epoch-1:
                print("pg_epoch: ", pg_epoch)
                print("avg pg_epo_loss: ", pg_epo_loss / len(train_loader))

                with torch.no_grad():
                    loss = 0
                    for x, y in valid_loader:
                        aug_encoding = torch.Tensor(x).long()
                        pg_val = torch.Tensor(y).float()
                        if torch.cuda.is_available():
                            aug_encoding, pg_val = aug_encoding.cuda(), pg_val.cuda()
                        ensemble_np_lst[e_np_i].eval()
                        pred = ensemble_np_lst[e_np_i](aug_encoding)
                        loss += loss_fn(pred, pg_val).item() * aug_encoding.shape[0]
                        del aug_encoding, pg_val, pred

                    print("avg val_loss: ", loss / len(valid_loader))

    return


def get_neural_predictor_class_embedding_op_model(local_rank=-1, num_classes=100):
    from LA3.networks.neural_predictor_model import Neural_Predictor_Embedding_Class_Op_Model
    model = Neural_Predictor_Embedding_Class_Op_Model(num_cls=num_classes)

    if local_rank >= 0:
        device = torch.device('cuda', local_rank)
        model = model.to(device)
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    else:
        if torch.cuda.is_available():
            model = model.cuda()

    return model


class mRMR:

    def __init__(self, inputs, seed, alpha, metric='manhattan'):
        self.features = inputs[-1]
        self.scores = np.expand_dims(np.max(inputs[1]) - inputs[1], -1)
        self.metric = metric
        self.alpha = alpha
        self.num = self.features.shape[0]
        self.already_selected = []
        # self.weights = (np.max(self.scores) - self.scores) / (np.max(self.scores) - np.min(self.scores))
        np.random.seed(seed)

    def update_distance(self, cluster_centers, only_new=True, reset_dist=False):
        mean_score = np.mean(self.scores)
        mean_score = mean_score if mean_score != 0 else 1
        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers if d not in self.already_selected]
        if cluster_centers:
            x = self.features[cluster_centers]
            dist = pairwise_distances(self.features, x, metric=self.metric)

            if self.min_distances is None:
                self.min_distances = self.scores + self.alpha * mean_score * np.mean(dist, axis=1).reshape(-1, 1)
                # self.min_distances = 100 * np.mean(dist, axis=1).reshape(-1, 1)
                self.min_distances[self.already_selected] = -np.inf
            else:
                self.min_distances = self.scores + self.alpha * mean_score * np.mean(dist, axis=1).reshape(-1, 1)
                self.min_distances[self.already_selected] = -np.inf

    def select_batch_(self, already_selected, N):
        try:
            self.update_distance(already_selected, only_new=False, reset_dist=True)
        except:
            self.update_distance(already_selected, only_new=True, reset_dist=False)

        new_batch = already_selected

        for _ in range(N):
            if len(self.already_selected) == 0:
                ind = np.random.choice(np.arange(self.num))
            else:
                ind = np.argmax(self.min_distances)
            assert ind not in self.already_selected

            self.already_selected.append(ind)
            new_batch.append(ind)
            self.update_distance(new_batch, only_new=False, reset_dist=True)

        print('Maximum distance from cluster centers is %0.2f' % max(self.min_distances))

        self.already_selected = already_selected

        return new_batch


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')

    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--num_class',)
    args = parser.parse_args()

    import time
    t = time.time()

    # The aug policy index to aug policy dict; The aug policy index to one-hot encoding
    aug_policy_index_dict, aug_policy_encoding_dict, aug_policy_get_index_dict = get_total_augment_policy_op(for_autoaug=False)

    ensemble_neural_predictor = []
    ensemble_np_optimizer = []

    aug_pg_data = pickle.load(open(args.data_file, 'rb'))

    num_classes = len(set([d['cls_idx'] for d in aug_pg_data]))

    for np_idx in range(5):
        np_model = get_neural_predictor_class_embedding_op_model(local_rank=args.local_rank, num_classes=num_classes)
        ensemble_neural_predictor.append(np_model)

        np_optimizer = optim.Adam(
            np_model.parameters(),
            lr=0.01,
            betas=(0.9, 0.999)
        )
        ensemble_np_optimizer.append(np_optimizer)

    aug_op_lst = augment_list_name_plus_identity(for_autoaug=False)

    input_tensor = []
    for l in aug_pg_data:
        aug1, aug2, aug3 = aug_policy_index_dict[l['aug_idx']]
        aug1 = aug_op_lst.index(aug1)
        aug2 = aug_op_lst.index(aug2)
        aug3 = aug_op_lst.index(aug3)
        rwd = l['pg_val_cls']
        input_tensor.append([aug1, aug2, aug3, l['cls_idx'], rwd])
        if [aug_op_lst[aug1], aug_op_lst[aug2], aug_op_lst[aug3]].count('Invert') > 1:
            print(f'{aug_op_lst[aug1]}, {aug_op_lst[aug2]}, {aug_op_lst[aug3]}: {rwd}')


    # Train neural predictor ensemble
    train_neural_predictors_batch(ensemble_neural_predictor, ensemble_np_optimizer, 100, input_tensor) # debug

    policies = list(aug_policy_get_index_dict.keys())

    candidate_aug_policy_lst = {c: [] for c in range(num_classes)}
    top_k_candidate_aug_policy_lst = {c: [] for c in range(num_classes)}


    with torch.no_grad():
        for c in range(num_classes):
            for aug_p_idx in policies:
                candidate_aug_policy_dict = dict()
                candidate_aug_policy_dict['policy'] = aug_policy_get_index_dict[aug_p_idx]
                aug1, aug2, aug3 = aug_p_idx
                aug1 = aug_op_lst.index(aug1)
                aug2 = aug_op_lst.index(aug2)
                aug3 = aug_op_lst.index(aug3)
                candidate_aug_policy_dict['encoding'] = [aug1, aug2, aug3, c]
                candidate_aug_policy_lst[c].append(candidate_aug_policy_dict)


            # Tensor of encodings of aug candidates
            aug_policy_candidates_enc_tensor = torch.from_numpy(np.array([cand['encoding'] for cand in candidate_aug_policy_lst[c]]))
            if torch.cuda.is_available():
                aug_policy_candidates_enc_tensor = aug_policy_candidates_enc_tensor.cuda()

            aug_policy_candidates_pg_predictions = []
            aug_policy_candidates_pg_embeddings = []

            # Predict the prediction gain of the candidate aug policies
            for e_np_i in range(len(ensemble_neural_predictor)):
                # Predict the prediction gain of the candidate aug policies
                ensemble_neural_predictor[e_np_i].eval()

                this_np_candidates_preds = ensemble_neural_predictor[e_np_i](aug_policy_candidates_enc_tensor.long())

                aug_policy_candidates_pg_predictions.append(this_np_candidates_preds.cpu().squeeze().tolist())

                del this_np_candidates_preds

            embed = np.zeros((aug_policy_candidates_enc_tensor.shape[0], len(aug_op_lst)))
            for i, item in enumerate(candidate_aug_policy_lst[c]):
                aug1, aug2, aug3, c = item['encoding']
                embed[i][aug1] += 1
                embed[i][aug2] += 1
                embed[i][aug3] += 1

            del aug_policy_candidates_enc_tensor

            pred = np.mean(aug_policy_candidates_pg_predictions, axis=0)
            # embed = np.mean(aug_policy_candidates_pg_embeddings, axis=0)
            top_k_candidate_aug_policy_lst[c].extend([[candidate_aug_policy_lst[c][idx]['policy'] for idx in range(pred.shape[0])], pred, embed])

    selected_policies = {c: [] for c in range(num_classes)}

    for c in range(num_classes):
        sampler = mRMR(top_k_candidate_aug_policy_lst[c], seed=1234+c, alpha=args.alpha)
        max_idx = top_k_candidate_aug_policy_lst[c][1].argmax()
        selected = sampler.select_batch_([max_idx], args.topk)
        selected_policies[c] = [policies[i] for i in selected]
        # break #debug

    for aug in augment_list_name_plus_identity(for_autoaug=False):
        print(f"{aug}: {[aug for augs in selected_policies[0] for aug in augs].count(aug) / (3*len(selected_policies[0]))}")

    print('-----------------------------------')

    for c in range(num_classes):
        idxs = top_k_candidate_aug_policy_lst[c][0]
        preds = top_k_candidate_aug_policy_lst[c][1]
        aug_score_dict = {aug: [] for aug in augment_list_name_plus_identity(False)}
        for i in idxs:
            aug1, aug2, aug3 = aug_policy_index_dict[i]
            aug_score_dict[aug1].append(preds[i])
            aug_score_dict[aug2].append(preds[i])
            aug_score_dict[aug3].append(preds[i])
        for aug, scores in aug_score_dict.items():
            print(f'{aug}: {np.mean(scores)}')
        break

    # if not os.path.exists(args.data_file.split('/')[0] + '/policies'):
    #     os.mkdir(args.data_file.split('/')[0] + '/policies')

    with open('./policies/selected_%d_alpha_%.2f_numpred_%d_policy.pkl' % (args.topk, args.alpha, 5), 'wb') as f:
    # with open(args.data_file.split('/')[0] + '/policies/selected_%d_alpha_%.2f_numpred_%d_policy.pkl' % (args.topk, args.alpha, 5), 'wb') as f:
        pickle.dump(selected_policies, f)

    elapsed = time.time() - t

    print('elapsed time (hours): ', (elapsed / 3600.))

    logger.info('done.')
    logger.info('model: %s' % C.get()['model'])
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))

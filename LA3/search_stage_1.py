import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

import logging
import math
import os

from torch import nn, optim
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from tqdm import tqdm
from theconf import Config as C, ConfigArgumentParser

from LA3.common import get_logger, add_filehandler
from LA3.data import get_dataloaders_search, custDatasetGivenClsIdxOpAug
from LA3.metrics import accuracy, Accumulator, CrossEntropyLabelSmooth, class_recall
from LA3.networks import get_model, num_class, get_neural_predictor_class_embedding_op_model
from LA3.aug_mixup import CrossEntropyMixUpLabelSmooth, mixup

from LA3.augmentations import *
import random
import pickle as pickle




logger = get_logger('LA3')
logger.setLevel(logging.INFO)


def run_epoch(model, loader, loss_fn, optimizer, desc_default='', epoch=0, writer=None, verbose=0, scheduler=None, is_master=True, ema=None, wd=0.0, tqdm_disabled=False):
    # num_classes = len(validloader.dataset.dataset.dataset.classes)
    if verbose:
        loader = tqdm(loader, disable=tqdm_disabled)
        loader.set_description('[%s %04d/%04d]' % (desc_default, epoch, C.get()['epoch']))

    params_without_bn = [params for name, params in model.named_parameters() if not ('_bn' in name or '.bn' in name)]

    loss_ema = None
    metrics = Accumulator()
    cnt = 0
    total_steps = len(loader)
    steps = 0
    class_res = {i: [0, 0] for i in range(num_classes)}
    for data, label in loader:
        steps += 1
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()


        if C.get().conf.get('mixup', 0.0) <= 0.0 or optimizer is None:
            preds = model(data)
            loss = loss_fn(preds, label)
        else:   # mixup
            data, targets, shuffled_targets, lam = mixup(data, label, C.get()['mixup'])
            preds = model(data)
            loss = loss_fn(preds, targets, shuffled_targets, lam)
            del shuffled_targets, lam

        if optimizer:
            loss += wd * (1. / 2.) * sum([torch.sum(p ** 2) for p in params_without_bn])
            loss.backward()
            grad_clip = C.get()['optimizer'].get('clip', 5.0)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            if ema is not None:
                ema(model, (epoch - 1) * total_steps + steps)

        top1, top5 = accuracy(preds, label, (1, 5))
        class_res = class_recall(preds, label, class_res)
        metrics.add_dict({
            'loss': loss.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })
        cnt += len(data)
        if loss_ema:
            loss_ema = loss_ema * 0.9 + loss.item() * 0.1
        else:
            loss_ema = loss.item()
        if verbose:
            postfix = metrics / cnt
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            postfix['loss_ema'] = loss_ema
            loader.set_postfix(postfix)

        if scheduler is not None:
            scheduler.step(epoch - 1 + float(steps) / total_steps)

        del preds, loss, top1, top5, data, label

    for key in class_res.keys():
        class_res[key] = class_res[key][0] / class_res[key][1] if class_res[key][1] != 0 else 0
    if tqdm_disabled and verbose:
        if optimizer:
            logger.info('[%s %03d/%03d] %s lr=%.6f', desc_default, epoch, C.get()['epoch'], metrics / cnt, optimizer.param_groups[0]['lr'])
        else:
            logger.info('[%s %03d/%03d] %s', desc_default, epoch, C.get()['epoch'], metrics / cnt)

    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    if verbose:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
    return metrics, class_res


def generate_candidates(aug_policy_idx_lst, cls, C_num=100):
    # Return a list of generated aug policy dicts
    candidates = []

    # Record the generated aug policy's index for preventing duplicates
    candidates_policy_idx_dict = dict()

    aug_op_lst = augment_list_name_plus_identity(False)
    aug_op_mag_lst = list(range(10))

    for aug_policy_idx in aug_policy_idx_lst:
        # Get the aug operation (op1, mag1, op2, mag2)
        selected_aug_op = aug_policy_index_dict[aug_policy_idx]

        op1_name = selected_aug_op[0]
        op2_name = selected_aug_op[1]

        # Generate 5 new policies for this op pair
        for i in range(5):
            op3_name = random.choice(aug_op_lst)
            candidate_aug_policy = (op1_name, op2_name, op3_name)

            candidate_aug_policy_idx = aug_policy_get_index_dict[candidate_aug_policy]

            if candidate_aug_policy_idx not in candidates_policy_idx_dict:
                candidate_aug_policy_dict = dict()
                candidate_aug_policy_dict['aug_idx'] = candidate_aug_policy_idx
                candidate_aug_policy_dict['encoding'] = [aug_op_lst.index(op1_name), aug_op_lst.index(op2_name), aug_op_lst.index(op3_name), cls]
                candidates.append(candidate_aug_policy_dict)

                candidates_policy_idx_dict[candidate_aug_policy_idx] = 1

        # Generate 5 new policies for this op pair
        for i in range(5):
            op2_name = random.choice(aug_op_lst)
            op3_name = random.choice(aug_op_lst)
            candidate_aug_policy = (op1_name, op2_name, op3_name)

            candidate_aug_policy_idx = aug_policy_get_index_dict[candidate_aug_policy]

            if candidate_aug_policy_idx not in candidates_policy_idx_dict:
                candidate_aug_policy_dict = dict()
                candidate_aug_policy_dict['aug_idx'] = candidate_aug_policy_idx
                candidate_aug_policy_dict['encoding'] = [aug_op_lst.index(op1_name), aug_op_lst.index(op2_name),
                                                         aug_op_lst.index(op3_name), cls]
                candidates.append(candidate_aug_policy_dict)

                candidates_policy_idx_dict[candidate_aug_policy_idx] = 1

    # 50 generated policies by sampling op from collected op score and random magnitude
    zero_play_aug_policy = np.where(aug_policy_cls_played_num[:,cls] == 0)[0]
    num = min(50, len(zero_play_aug_policy))
    zero_play_cand = np.random.choice(zero_play_aug_policy, num)
    for idx in zero_play_cand:
        if idx not in candidates_policy_idx_dict:
            op1_name, op2_name, op3_name = aug_policy_index_dict[idx]
            candidate_aug_policy_dict = dict()
            candidate_aug_policy_dict['aug_idx'] = idx
            candidate_aug_policy_dict['encoding'] = [aug_op_lst.index(op1_name), aug_op_lst.index(op2_name),
                                                     aug_op_lst.index(op3_name), cls]
            candidates.append(candidate_aug_policy_dict)

            candidates_policy_idx_dict[idx] = 1

    # the ucb value of aug op pairs: avg validation acc + exploration val
    ucb_arr = (100.0 - (aug_op_pair_cls_score[c] / aug_op_pair_cls_played_num[c])) + np.sqrt(2*np.log(total_aug_cls_played_num[c]) / aug_op_pair_cls_played_num[c])
    ucb_arr = np.nan_to_num(ucb_arr)

    for i in range(C_num - len(candidates)):
        # Randomly sample aug op pair according to ucb_arr
        candidate_op_pair_idx = np.random.choice(np.arange(len(ucb_arr)), p = ucb_arr / ucb_arr.sum())

        candidate_op1_name = idx_aug_op_pair_dict[candidate_op_pair_idx][0]
        candidate_op2_name = idx_aug_op_pair_dict[candidate_op_pair_idx][1]
        candidate_op3_name = idx_aug_op_pair_dict[candidate_op_pair_idx][2]

        candidate_aug_policy = (candidate_op1_name, candidate_op2_name, candidate_op3_name)

        candidate_aug_policy_idx = aug_policy_get_index_dict[candidate_aug_policy]

        if candidate_aug_policy_idx not in candidates_policy_idx_dict:
            candidate_aug_policy_dict = dict()
            candidate_aug_policy_dict['aug_idx'] = candidate_aug_policy_idx
            candidate_aug_policy_dict['encoding'] = [aug_op_lst.index(candidate_op1_name), aug_op_lst.index(candidate_op2_name),
                                                     aug_op_lst.index(candidate_op3_name), cls]

            candidates.append(candidate_aug_policy_dict)

            candidates_policy_idx_dict[candidate_aug_policy_idx] = 1

    return candidates


def train_neural_predictors(ensemble_np_lst, ensemble_np_opt_lst, total_epoch, aug_pg_data):

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


    data = list(zip([d[:5] for d in aug_pg_data], [d[-1:] for d in aug_pg_data]))
    random.shuffle(data)
    train_num = int(len(data) * 0.9)
    train_loader = CustomLoader(data[:train_num], 10000)
    valid_loader = CustomLoader(data[train_num:], 10000)

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


def get_model_and_base_performance(tag, validloader, reporter=None, save_path=None, local_rank=-1):

    is_master = local_rank < 0 or dist.get_rank() == 0
    if is_master:
        add_filehandler(logger, args.pretrained_model + '.log')

    if not reporter:
        reporter = lambda **kwargs: 0

    # create a model & an optimizer
    model = get_model(C.get()['model'], num_class(C.get()['dataset']), local_rank=local_rank)
    model_ema = get_model(C.get()['model'], num_class(C.get()['dataset']), local_rank=-1)
    model_ema.eval()

    criterion_ce = criterion = CrossEntropyLabelSmooth(num_class(C.get()['dataset']), C.get().conf.get('lb_smooth', 0))
    if C.get().conf.get('mixup', 0.0) > 0.0:
        criterion = CrossEntropyMixUpLabelSmooth(num_class(C.get()['dataset']), C.get().conf.get('lb_smooth', 0))

    if not tag or not is_master:
        from LA3.metrics import SummaryWriterDummy as SummaryWriter
        logger.warning('tag not provided, no tensorboard log.')
    else:
        from tensorboardX import SummaryWriter
    writers = [SummaryWriter(log_dir='./logs/%s/%s' % (tag, x)) for x in ['train', 'test', 'val']]

    tqdm_disabled = bool(os.environ.get('TASK_NAME', '')) and local_rank != 0  # KakaoBrain Environment

    # Load the model and optimizer weights from pre-trained randomly augmented model
    if save_path and os.path.exists(save_path):
        logger.info('%s file found. loading...' % save_path)
        data = torch.load(save_path)
        key = 'model' if 'model' in data else 'state_dict'

        if 'epoch' not in data:
            model.load_state_dict(data)
        else:
            logger.info('checkpoint epoch@%d' % data['epoch'])
            if not isinstance(model, (DataParallel, DistributedDataParallel)):
                model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
            else:
                model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
        del data
    else:
        raise ValueError('invalid pre-trained model path=%s' % save_path)

    model.eval()
    rs = dict()
    with torch.no_grad():
        rs['valid'], valid_cls_res = run_epoch(model, validloader, criterion_ce, None, desc_default='*valid', epoch=1,
                                              writer=writers[1], verbose=is_master, tqdm_disabled=tqdm_disabled)

    return model, valid_cls_res


def eval_with_auged_data(model, tag, dataroot, trainloader, test_ratio=0.0, cv_fold=0, reporter=None, metric='last', save_path=None, only_eval=False, local_rank=-1, evaluation_interval=1, is_init=True, given_aug_idx=None, collected_score_aug_idx=None):
    global pre_trained_res
    global aug_op_pair_score
    global aug_op_pair_cls_score
    global aug_op_pair_played_num
    global aug_op_pair_cls_played_num
    global total_aug_played_num

    total_batch = C.get()["batch"]
    if local_rank >= 0:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=int(os.environ['WORLD_SIZE']))
        device = torch.device('cuda', local_rank)
        torch.cuda.set_device(device)

        C.get()['lr'] *= dist.get_world_size()
        logger.info(f'local batch={C.get()["batch"]} world_size={dist.get_world_size()} ----> total batch={C.get()["batch"] * dist.get_world_size()}')
        total_batch = C.get()["batch"] * dist.get_world_size()

    is_master = local_rank < 0 or dist.get_rank() == 0
    if is_master:
        add_filehandler(logger, args.pretrained_model + '.log')

    if not reporter:
        reporter = lambda **kwargs: 0

    criterion_ce = criterion = CrossEntropyLabelSmooth(num_class(C.get()['dataset']), C.get().conf.get('lb_smooth', 0))
    if C.get().conf.get('mixup', 0.0) > 0.0:
        criterion = CrossEntropyMixUpLabelSmooth(num_class(C.get()['dataset']), C.get().conf.get('lb_smooth', 0))

    if not tag or not is_master:
        from LA3.metrics import SummaryWriterDummy as SummaryWriter
        logger.warning('tag not provided, no tensorboard log.')
    else:
        from tensorboardX import SummaryWriter
    writers = [SummaryWriter(log_dir='./logs/%s/%s' % (tag, x)) for x in ['train', 'test', 'val']]

    epoch_start = 1

    if local_rank >= 0:
        for name, x in model.state_dict().items():
            dist.broadcast(x, 0)
        logger.info(f'multinode init. local_rank={dist.get_rank()} is_master={is_master}')
        torch.cuda.synchronize()

    tqdm_disabled = bool(os.environ.get('TASK_NAME', '')) and local_rank != 0  # KakaoBrain Environment

    # Get val result after training one batch or have finetuned 3 epochs
    if given_aug_idx is not None:

        model.eval()

        with torch.no_grad():
            val_results, val_cls_res = run_epoch(model, trainloader, criterion_ce, None, desc_default='*val', epoch=epoch_start, writer=writers[2], verbose=is_master, tqdm_disabled=tqdm_disabled)

            pre_acc = sum(pre_trained_res.values()) / len(pre_trained_res)
            collected_score = round(100.0*(pre_acc-val_results["top1"]), 4)


            aug_policy_score[collected_score_aug_idx] += collected_score
            aug_policy_played_num[collected_score_aug_idx] += 1
            for c, aug_idx in enumerate(collected_score_aug_idx):
                aug_policy_cls_score[aug_idx][c] += round(100.0*(pre_trained_res[c]-val_cls_res[c]), 4)
                aug_policy_cls_played_num[aug_idx][c] += 1
                # if c == 0:
                #     print(aug_policy_index_dict[aug_idx])
                #     print(round(100.0*(pre_trained_res[c]-val_cls_res[c]), 4))

            for c, aug_idx in enumerate(collected_score_aug_idx):
                # Get the op name of the selected aug idx
                collected_score_aug_policy = aug_policy_index_dict[aug_idx]

                # Update the collected score for aug op pair
                aug_pair = collected_score_aug_policy
                aug_op_pair_cls_score[c][aug_op_pair_idx_dict[aug_pair]] += collected_score
                aug_op_pair_cls_played_num[c][aug_op_pair_idx_dict[aug_pair]] += 1
                total_aug_cls_played_num[c] += 1



def eval_neural_predictor(iter_num='', num_classes=100):
    # Create an ensemble of neural predictors, and corresponding optimizers
    ensemble_neural_predictor = []
    ensemble_np_optimizer = []

    for np_idx in range(5):
        np_model = get_neural_predictor_class_embedding_op_model(local_rank=-1, num_class=num_classes)
        ensemble_neural_predictor.append(np_model)

        np_optimizer = optim.Adam(
            np_model.parameters(),
            lr=0.01,
            betas=(0.9, 0.999)
        )
        ensemble_np_optimizer.append(np_optimizer)

    aug_pg_data = []
    temp_aug_score = {c: [] for c in range(num_classes)}
    temp_aug_idx = {c: [] for c in range(num_classes)}

    for aug_idx in range(len(aug_policy_index_dict)):
        for c in range(num_classes):
            if aug_policy_cls_played_num[aug_idx][c] != 0:
                eval_aug_pg_val = dict()
                eval_aug_pg_val['aug_idx'] = aug_idx
                # eval_aug_pg_val['encoding'] = aug_policy_encoding_dict[aug_idx]
                eval_aug_pg_val['cls_idx'] = c

                eval_aug_pg_val['pg_val'] = aug_policy_score[aug_idx] / aug_policy_played_num[aug_idx]
                eval_aug_pg_val['pg_val_cls'] = aug_policy_cls_score[aug_idx][c] / aug_policy_cls_played_num[aug_idx][c]

                aug_pg_data.append(eval_aug_pg_val)
                temp_aug_score[c].append(eval_aug_pg_val['pg_val_cls'])
                temp_aug_idx[c].append(eval_aug_pg_val['aug_idx'])

    direc = C.get()['dataset'] + '_' + C.get()['model']['type'] + '/'
    if not os.path.exists(direc):
        os.mkdir(direc)

    with open(direc + 'aug_pg_data.pkl', 'wb') as handle:
        pickle.dump(aug_pg_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    input_tensor = []
    for l in aug_pg_data:
        aug1, aug2, aug3 = aug_policy_index_dict[l['aug_idx']]
        aug1 = aug_op_lst.index(aug1)
        aug2 = aug_op_lst.index(aug2)
        aug3 = aug_op_lst.index(aug3)
        rwd = l['pg_val_cls']
        input_tensor.append([aug1, aug2, aug3, l['cls_idx'], rwd])
    # Train the prediction gain data
    train_neural_predictors(ensemble_neural_predictor, ensemble_np_optimizer, 100, input_tensor)

    # Evaluate all aug policies in the search space
    candidate_aug_policy_lst = {c:[] for c in range(num_classes)}

    for c in range(num_classes):
        for aug_p_idx in range(len(aug_policy_index_dict)):
            candidate_aug_policy_dict = dict()
            candidate_aug_policy_dict['aug_idx'] = aug_p_idx
            aug1, aug2, aug3 = aug_policy_index_dict[aug_p_idx]
            aug1 = aug_op_lst.index(aug1)
            aug2 = aug_op_lst.index(aug2)
            aug3 = aug_op_lst.index(aug3)
            candidate_aug_policy_dict['encoding'] = [aug1, aug2, aug3, c]

            candidate_aug_policy_lst[c].append(candidate_aug_policy_dict)

        # Tensor of encodings of aug candidates
        aug_policy_candidates_enc_tensor = torch.from_numpy(np.array([cand['encoding'] for cand in candidate_aug_policy_lst[c]]))
        aug_policy_candidates_enc_tensor = aug_policy_candidates_enc_tensor.cuda()

        aug_policy_candidates_pg_predictions = []

        for e_np_i in range(len(ensemble_neural_predictor)):
            # Predict the prediction gain of the candidate aug policies
            ensemble_neural_predictor[e_np_i].eval()

            this_np_candidates_preds = ensemble_neural_predictor[e_np_i](aug_policy_candidates_enc_tensor)

            aug_policy_candidates_pg_predictions.append(this_np_candidates_preds.cpu().squeeze().tolist())

            del this_np_candidates_preds

        del aug_policy_candidates_enc_tensor


def get_aug_op_triple_index_dict():
    aug_op_lst = augment_list_name_plus_identity(False)

    aug_op_pair_idx_dict = dict()
    idx_aug_op_pair_dict = dict()

    op_pair_idx = 0

    for op1 in aug_op_lst:
        for op2 in aug_op_lst:
            for op3 in aug_op_lst:
                aug_op_pair_idx_dict[(op1, op2, op3)] = op_pair_idx
                idx_aug_op_pair_dict[op_pair_idx] = (op1, op2, op3)

                op_pair_idx += 1

    return aug_op_pair_idx_dict, idx_aug_op_pair_dict


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='/data/private/pretrainedmodels', help='torchvision data folder')
    parser.add_argument('--pretrained_model', type=str, default='test.pth')
    parser.add_argument('--cv-ratio', type=float, default=0.0)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--evaluation-interval', type=int, default=1)
    parser.add_argument('--only-eval', action='store_true')
    parser.add_argument('--N', type=int, default=500)

    args = parser.parse_args()

    assert (args.only_eval and args.pretrained_model) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'

    path = C.get()['dataset'] + '_' + C.get()['model']['type'] + '/'
    if not os.path.exists(path):
        os.mkdir(path)

    if not args.only_eval:
        if args.pretrained_model:
            logger.info('checkpoint will be saved at %s' % args.pretrained_model)
        else:
            logger.warning('Provide --save argument to save the checkpoint. Without it, training result will not be saved!')

    import time
    t = time.time()

    total_trainset, validloader, testloader_, transform_train_unnorm, transform_train_norm_op = get_dataloaders_search(C.get()['dataset'], C.get()['batch'], args.dataroot)
    valid_set = validloader.dataset

    num_classes = len(total_trainset.dataset.classes)

    model, pre_trained_res = get_model_and_base_performance(args.tag, validloader, save_path=args.pretrained_model, local_rank=args.local_rank)
    # args.pretrained_model = path + args.pretrained_model

    valid_set.dataset.transform = None

    # The aug policy index to aug policy dict; The aug policy index to one-hot encoding
    aug_policy_index_dict, aug_policy_encoding_dict, aug_policy_get_index_dict = get_total_augment_policy_op(for_autoaug=False)

    # Collected Aug policy scores
    aug_policy_score = np.zeros(len(aug_policy_index_dict))
    # Aug policy played number
    aug_policy_played_num = np.zeros(len(aug_policy_index_dict))
    # Aug policy scores for each class
    aug_policy_cls_score = np.zeros((len(aug_policy_index_dict), num_classes))
    # Aug policy played number for each class
    aug_policy_cls_played_num = np.zeros((len(aug_policy_index_dict), num_classes))


    # get aug op pairs to index dict
    aug_op_pair_idx_dict, idx_aug_op_pair_dict = get_aug_op_triple_index_dict()

    # total op pair scores
    aug_op_pair_score = np.zeros(len(aug_op_pair_idx_dict))
    # aug op pair played number
    aug_op_pair_played_num = np.zeros(len(aug_op_pair_idx_dict))
    # total played num
    total_aug_played_num = 0

    aug_op_pair_cls_score = np.zeros((num_classes, len(aug_op_pair_idx_dict)))
    aug_op_pair_cls_played_num = np.zeros((num_classes, len(aug_op_pair_idx_dict)))
    total_aug_cls_played_num = {c:0 for c in range(num_classes)}

    init_task_idx_lst = [[] for c in range(num_classes)]

    aug_op_lst = augment_list_name_plus_identity(for_autoaug=False)

    for c in range(num_classes):
        for op_pair_val in aug_op_pair_idx_dict:
            init_task_idx_lst[c].append(aug_policy_get_index_dict[op_pair_val])
        random.shuffle(init_task_idx_lst[c])
        init_task_idx_lst[c] = init_task_idx_lst[c][:100]

    init_task_idx_lst = np.array(init_task_idx_lst).T

    # Get each init aug policy score
    for aug_idx in init_task_idx_lst:
        aug_validset = custDatasetGivenClsIdxOpAug(valid_set, transform_train_unnorm,
                                                   transform_train_norm_op, aug_idx)
        aug_validloader = torch.utils.data.DataLoader(aug_validset, batch_size=C.get()['batch'], shuffle=False,
                                                      num_workers=8, pin_memory=True, drop_last=False)

        eval_with_auged_data(model, args.tag, args.dataroot, aug_validloader, test_ratio=args.cv_ratio, cv_fold=args.cv, save_path=args.pretrained_model, only_eval=args.only_eval, local_rank=args.local_rank, metric='test', evaluation_interval=args.evaluation_interval, is_init=False, given_aug_idx=aug_idx, collected_score_aug_idx=aug_idx)
        # break # debug

    print("Init done")

    # The list of top k candidate aug policies
    top_k_candidate_aug_policy_lst = []

    # Collected 2500 aug scores (finetune 10 epochs)
    for i in range(args.N):
        if len(top_k_candidate_aug_policy_lst) == 0:
            print("iter i: ", i)
            # Create an ensemble of neural predictors, and corresponding optimizers
            ensemble_neural_predictor = []
            ensemble_np_optimizer = []

            for np_idx in range(5):
                np_model = get_neural_predictor_class_embedding_op_model(local_rank=args.local_rank, num_class=num_classes)
                ensemble_neural_predictor.append(np_model)

                np_optimizer = optim.Adam(
                    np_model.parameters(),
                    lr=0.01,
                    betas=(0.9, 0.999)
                )
                ensemble_np_optimizer.append(np_optimizer)

            aug_pg_data = []

            temp_aug_score = {c:[] for c in range(num_classes)}
            temp_aug_idx = {c:[] for c in range(num_classes)}

            for aug_idx in range(len(aug_policy_index_dict)):
                for c in range(num_classes):
                    if aug_policy_cls_played_num[aug_idx][c] != 0:
                        eval_aug_pg_val = dict()
                        eval_aug_pg_val['aug_idx'] = aug_idx
                        # eval_aug_pg_val['encoding'] = aug_policy_encoding_dict[aug_idx]
                        eval_aug_pg_val['cls_idx'] = c

                        eval_aug_pg_val['pg_val'] = aug_policy_score[aug_idx] / aug_policy_played_num[aug_idx]
                        eval_aug_pg_val['pg_val_cls'] = aug_policy_cls_score[aug_idx][c] / aug_policy_cls_played_num[aug_idx][c]

                        aug_pg_data.append(eval_aug_pg_val)
                        temp_aug_score[c].append(eval_aug_pg_val['pg_val_cls'])
                        temp_aug_idx[c].append(eval_aug_pg_val['aug_idx'])

            input_tensor = []
            for l in aug_pg_data:
                aug1, aug2, aug3 = aug_policy_index_dict[l['aug_idx']]
                aug1 = aug_op_lst.index(aug1)
                aug2 = aug_op_lst.index(aug2)
                aug3 = aug_op_lst.index(aug3)
                rwd = l['pg_val_cls']
                input_tensor.append([aug1, aug2, aug3, l['cls_idx'], rwd])

            # Train neural predictor ensemble
            train_neural_predictors(ensemble_neural_predictor, ensemble_np_optimizer, 100, input_tensor) # debug


            selected_aug_idx_arr = {c:[] for c in range(num_classes)}
            aug_policy_candidates = {c:[] for c in range(num_classes)}
            for c in range(num_classes):
                temp_aug_score_sorted = np.argsort(temp_aug_score[c])
                selected_aug_idx_arr[c] = np.array(temp_aug_idx[c])[temp_aug_score_sorted[:5]]

                aug_policy_candidates[c] = generate_candidates(list(selected_aug_idx_arr[c]), cls=c)

                # Tensor of encodings of aug candidates
                aug_policy_candidates_enc_tensor = torch.from_numpy(np.array([cand['encoding'] for cand in aug_policy_candidates[c]]))
                aug_policy_candidates_enc_tensor = aug_policy_candidates_enc_tensor.cuda()

                aug_policy_candidates_pg_predictions = []

                for e_np_i in range(len(ensemble_neural_predictor)):
                    # Predict the prediction gain of the candidate aug policies
                    ensemble_neural_predictor[e_np_i].eval()

                    this_np_candidates_preds = ensemble_neural_predictor[e_np_i](aug_policy_candidates_enc_tensor)

                    aug_policy_candidates_pg_predictions.append(this_np_candidates_preds.cpu().squeeze().tolist())

                    del this_np_candidates_preds

                del aug_policy_candidates_enc_tensor

                # Compute the acquisition function for all the candidate aug policies
                mean = np.mean(aug_policy_candidates_pg_predictions, axis=0)
                sorted_indices = np.argsort(mean)
                selected_indices = sorted_indices[:10]
                random.shuffle(selected_indices)

                if c == 0:
                    print('prediction scores for selected')
                    for idx in selected_indices:
                        score = np.mean([scores[idx] for scores in aug_policy_candidates_pg_predictions])
                        print(aug_policy_index_dict[aug_policy_candidates[c][idx]['aug_idx']])
                        print(score)

                # Add top k aug policies to top_k_candidate_aug_policy_lst for evaluation
                top_k_candidate_aug_policy_lst.append([aug_policy_candidates[c][idx]['aug_idx'] for idx in selected_indices])

            top_k_candidate_aug_policy_lst = list(np.array(top_k_candidate_aug_policy_lst).T)

        eval_aug_idx = top_k_candidate_aug_policy_lst.pop()

        aug_validset = custDatasetGivenClsIdxOpAug(valid_set, transform_train_unnorm,
                                                   transform_train_norm_op, eval_aug_idx)
        aug_validloader = torch.utils.data.DataLoader(aug_validset, batch_size=C.get()['batch'], shuffle=False,
                                                      num_workers=8, pin_memory=True, drop_last=False)

        eval_with_auged_data(model, args.tag, args.dataroot, aug_validloader, test_ratio=args.cv_ratio, cv_fold=args.cv, save_path=args.pretrained_model, only_eval=args.only_eval, local_rank=args.local_rank, metric='test', evaluation_interval=args.evaluation_interval, is_init=False, given_aug_idx=eval_aug_idx, collected_score_aug_idx=eval_aug_idx)


        # break # debug
        if i == args.N - 1: # debug
            eval_neural_predictor(iter_num=str(i), num_classes=num_classes)

    eval_neural_predictor(num_classes=num_classes)

    elapsed = time.time() - t

    print('elapsed time (hours): ', (elapsed / 3600.))

    logger.info('done.')
    logger.info('model: %s' % C.get()['model'])
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info(args.pretrained_model)

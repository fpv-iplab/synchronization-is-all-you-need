import argparse
import random
from torch import optim
import pandas as pd
from utils import *
from dataset_oracle import *
from model import *


# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_c2f_ensemble_output(outp, weights):
    ensemble_prob = F.softmax(outp[0], dim=1) * weights[0] / sum(weights)

    for i, outp_ele in enumerate(outp[1]):
        upped_logit = F.upsample(outp_ele, size=outp[0].shape[-1], mode='linear', align_corners=True)
        ensemble_prob = ensemble_prob + F.softmax(upped_logit, dim=1) * weights[i + 1] / sum(weights)

    return ensemble_prob


################## Trainer (change loss)
class Trainer:
    def __init__(self):
        set_seed(seed)
        self.model = C2F_TCN(config.feature_size, config.num_class)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        assert self.num_classes > 0, "wrong class numbers"
        print('Model Size: {}'.format(sum(p.numel() for p in self.model.parameters())))
        self.es = EarlyStop(patience=args.patience)

    def train(self, save_dir, num_epochs):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
        best_score = -10000
        for epoch in range(num_epochs):
            correct, total, nums = 0, 0, 0
            epoch_loss, ce_loss, smooth_loss = 0.0, 0.0, 0.0
            for i, item in enumerate(train_loader):
                nums += 1
                samples = item[0].to(device).permute(0, 2, 1) 
                count = item[1].to(device) #
                labels = item[2].to(device) #

                src_mask = torch.arange(labels.shape[1], device=labels.device)[None, :] < count[:, None] #
                src_mask[labels == -100] = 0
                src_mask = src_mask.to(device) #
                src_msk_send = src_mask.to(torch.float32).to(device).unsqueeze(1) #
                outputs_list = self.model(samples)
                outputs_ensemble = get_c2f_ensemble_output(outputs_list, config.ensem_weights)
                outp_wo_softmax = torch.log(outputs_ensemble + 1e-10)  # log is necessary because ensemble gives softmax output
                ce_l = self.ce(outp_wo_softmax, labels)
                mse_l = 0.17 * torch.mean(torch.clamp(self.mse(outp_wo_softmax[:, :, 1:], outp_wo_softmax.detach()[:, :, :-1]),min=0, max=16) * src_msk_send[:, :, 1:]) #
                loss = ce_l + mse_l #
                ce_loss += ce_l.item()
                smooth_loss += mse_l.item() #
                epoch_loss += loss.item() #
                
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                pred = torch.argmax(outputs_ensemble, dim=1)
                correct += float(torch.sum((pred == labels) * src_mask).item()) #
                total += float(torch.sum(src_mask).item()) #

            scheduler.step()
            pr_str = "[epoch %d]: loss = %.3f, ce = %.3f, sm = %.3f, acc = %.3f" % \
                     (epoch + 1, epoch_loss / nums, ce_loss / nums, smooth_loss / nums, float(correct) / total)
            print(pr_str)
            
            if (epoch + 1) % 25 == 0:
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")

            test_score = self.test(epoch)
            if test_score > best_score:
                best_score = test_score
                torch.save(self.model.state_dict(), save_dir + "/epoch-best" + ".model")
                print("Save for the best model")
            if self.es.step(loss=None, acc=test_score, criterion=lambda x1, x2: x2):
                print("Early stop!")
                exit(0)

    def test(self, epoch):
        self.model.eval()
        epoch_loss, ce_loss, smooth_loss = 0.0, 0.0, 0.0
        nums = 0
        with torch.no_grad():
            correct, total, nums = 0, 0, 0
            for i, item in enumerate(test_loader):
                nums += 1
                samples = item[0].to(device).permute(0, 2, 1)
                count = item[1].to(device) #
                labels = item[2].to(device) #
                src_mask = torch.arange(labels.shape[1], device=labels.device)[None, :] < count[:, None]
                src_mask[labels == -100] = 0
                src_mask = src_mask.to(device)
                src_msk_send = src_mask.to(torch.float32).to(device).unsqueeze(1)

                outputs_list = self.model(samples)
                outputs_ensemble = get_c2f_ensemble_output(outputs_list, config.ensem_weights)
                outp_wo_softmax = torch.log(outputs_ensemble + 1e-10)
                ce_l = self.ce(outp_wo_softmax, labels)
                mse_l = 0.17 * torch.mean(
                    torch.clamp(self.mse(outp_wo_softmax[:, :, 1:], outp_wo_softmax.detach()[:, :, :-1]),
                                min=0, max=16) * src_msk_send[:, :, 1:])
                loss = ce_l + mse_l
                ce_loss += ce_l.item()
                smooth_loss += mse_l.item()
                epoch_loss += loss.item()

                labels_2 = labels[:,src_mask[0]]
                outputs_ensemble_2 = outputs_ensemble[:,:,src_mask[0]]

                count_2 = src_mask.sum().view(1)
                postprocessor(outputs_ensemble_2, item[5], labels_2, count_2) # item[5]

                pred = torch.argmax(outputs_ensemble, dim=1)
                correct += float(torch.sum((pred == labels) * src_mask).item())
                total += float(torch.sum(src_mask).item())



            pr_str = "***[epoch %d]***: loss = %.3f, ce = %.3f, sm = %.3f" % (epoch + 1, epoch_loss / nums, ce_loss / nums, smooth_loss / nums)
            print(pr_str)

            path = os.path.join(results_dir, "tempt")
            if not os.path.exists(path):
                os.mkdir(path)
            if not os.path.exists(path + '_gt'):
                os.mkdir(path + '_gt')
            postprocessor.dump_to_directory(path, '_gt')
            final_edit_score, acc, overlap_scores = calculate_mof(path + '_gt', path, config.back_gd)
            postprocessor.start()

            results = {}
            # action: standard metrics
            results['custom acc'] = correct 
            results['f1_10'] = overlap_scores[0]
            results['f1_25'] = overlap_scores[1]
            results['f1_50'] = overlap_scores[2]
            results['f_acc'] = acc
            results['edit'] = final_edit_score
            results['total_score'] = results['f1_10'] + results['f1_25'] + results['f1_50'] + results['f_acc'] + \
                                     results['edit']

            print("---[epoch %d]---: tst edit = %.2f, f1_10 = %.2f, f1_25 = %.2f, f1_50 = %.2f, acc = %.2f, total = %.2f, my_acc = %.2f "
                % (epoch + 1, results['edit'], results['f1_10'], results['f1_25'], results['f1_50'],
                   results['f_acc'], results['total_score'], float(results['custom acc'])/total))

        self.model.train()
        return results['total_score']

    def predict(self, model_dir, results_dir, eval_loader, postprocessor_eval):
        print("====================================PREDICT")
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-best" + ".model", map_location=device))

            for i, item in enumerate(eval_loader):
                samples = item[0].to(device).permute(0, 2, 1)
                count = item[1].to(device)
                labels = item[2].to(device)

                idx = labels != -100
                labels_2 = labels[:,idx[0]]

                outputs_list = self.model(samples)
                outputs_ensemble = get_c2f_ensemble_output(outputs_list, config.ensem_weights)

                outputs_ensemble_2 = outputs_ensemble[:,:,idx[0]]

                count_2 = idx.sum().view(1)

                postprocessor_eval(outputs_ensemble_2, item[5], labels_2, count_2, item[7].to(device), item[8],
                                   count_2.to(device))

            postprocessor_eval.dump_to_directory(results_dir)
            final_edit_score, acc, overlap_scores = calculate_mof(results_dir + '_gt', results_dir, config.back_gd)
            postprocessor_eval.start()

            results = {}
            # action: standard metrics
            results['f1_10'] = overlap_scores[0]
            results['f1_25'] = overlap_scores[1]
            results['f1_50'] = overlap_scores[2]
            results['f_acc'] = acc
            results['edit'] = final_edit_score
            results['total_score'] = results['f1_10'] + results['f1_25'] + results['f1_50'] + results['f_acc'] + \
                                     results['edit']

            print("---[Eval]---: tst edit = %.2f, f1_10 = %.2f, f1_25 = %.2f, f1_50 = %.2f, acc = %.2f, total = %.2f "
                % (results['edit'], results['f1_10'], results['f1_25'], results['f1_50'],
                   results['f_acc'], results['total_score']))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--feature_path', type=str, default='/mnt/data/zhanzhong/assembly/')
parser.add_argument('--dataset', default="assembly")
parser.add_argument('--split', default='train')  # or 'train_val'
parser.add_argument('--seed', default='42')
parser.add_argument('--test_aug', type=int, default=0)
parser.add_argument('--patience', type=int, default=50)
args = parser.parse_args()

seed = int(args.seed)
set_seed(seed)

VIEWS = [''] #insert view name, ego or exo

config = dotdict(
    epochs=100,
    dataset=args.dataset,
    feature_size=1024,
    gamma=0.5,
    step_size=200,
    split=args.split)

if args.dataset == "assembly":
    config.chunk_size = 1 
    config.max_frames_per_video = 600 
    config.learning_rate = 1e-5 
    config.weight_decay = 1e-4
    config.batch_size = 1
    config.num_class = 33 #insert here your class number
    config.back_gd = [] #['Background']
    config.ensem_weights = [1, 1, 1, 1, 1, 1]
    if args.action == 'predict':
        if int(args.test_aug):
            config.chunk_size = list(range(10, 31, 7))
            config.weights = np.ones(len(config.chunk_size))
        else:
            config.chunk_size = [1]
            print('chunk_size:', config.chunk_size)
            config.weights = [1]
else:
    print('not defined yet')
    exit(1)

TYPE = '/c2f_{}'.format(args.seed)
model_dir = "" + args.dataset + "/" + args.split + TYPE #output model path
results_dir = "" + args.dataset + "/" + args.split + TYPE #output results path

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

vid_list_path = "data/coarse-annotations/coarse_splits/" 
gt_path = "data/coarse-annotations/coarse_labels/" 
mapping_file = "data/coarse-annotations/actions.csv" 
features_path = args.feature_path

config.features_path = features_path
config.gt_path = gt_path
config.VIEWS = VIEWS

actions = pd.read_csv(mapping_file, header=0,
                      names=['action_id', 'verb_id', 'noun_id', 'action_cls', 'verb_cls', 'noun_cls'])
actions_dict, label_dict = dict(), dict()
for _, act in actions.iterrows():
    actions_dict[act['action_cls']] = int(act['action_id'])
    label_dict[int(act['action_id'])] = act['action_cls']

num_classes = len(actions_dict)
assert num_classes == config.num_class

############################dataloader
def _init_fn(worker_id):
    np.random.seed(int(seed))

###########################postprocessor 
postprocessor = PostProcess(config, label_dict, actions_dict, gt_path).to(device)

trainer = Trainer()
if args.action == "train":
    train_dataset = AugmentDataset(config, fold=args.split, fold_file_name=vid_list_path, actions_dict=actions_dict, zoom_crop=(0.5, 2))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True,
                                               pin_memory=True, num_workers=6, collate_fn=collate_fn_override,
                                               worker_init_fn=_init_fn)

    test_dataset = AugmentDataset(config, fold='val', fold_file_name=vid_list_path, actions_dict=actions_dict, zoom_crop=(0.5, 2))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False,
                                              pin_memory=True, num_workers=6, collate_fn=collate_fn_override,
                                              worker_init_fn=_init_fn)

    trainer.train(model_dir, num_epochs=config.epochs)

if args.action == "predict":
    eval_dataset = AugmentDataset_test(config, fold='val', fold_file_name=vid_list_path, actions_dict=actions_dict, chunk_size=config.chunk_size)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=config.batch_size, shuffle=False,
                                              pin_memory=True, num_workers=6, collate_fn=collate_fn_override_test,
                                              worker_init_fn=_init_fn)
    postprocessor_eval = PostProcess_test(config.weights, label_dict, actions_dict, gt_path).to(device)

    if not os.path.exists(os.path.join(results_dir, 'prediction{}'.format('_aug' if int(args.test_aug) else ''))):
        os.makedirs(os.path.join(results_dir, 'prediction{}'.format('_aug' if int(args.test_aug) else '')))
    trainer.predict(model_dir, os.path.join(results_dir, 'prediction{}'.format('_aug' if int(args.test_aug) else '')),
                    eval_loader, postprocessor_eval)


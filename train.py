import time
import logging
from data_builder import *
import argparse
from networks_for_CIFAR import *
from networks_for_ImageNet import *
from utils import accuracy, AvgrageMeter, save_checkpoint, get_model, create_para_dict, read_param, record_param, deletStrmodule, randomize_gate
import sys
sys.path.append("..")
from layers import *
from tensorboardX import SummaryWriter
from torch.cuda import amp
from schedulers import *
from Regularization import *
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import random

####################################################
# args                                             #
#                                                  #
####################################################

def get_args():
    parser = argparse.ArgumentParser("Gated Spiking Neural Networks")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval-resume', type=str, default='./raw/models', help='path for eval model')
    parser.add_argument('--train-resume', type=str, default='./raw/models', help='path for train model')
    parser.add_argument('--batch-size', type=int, default=40, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='total epochs used in training SuperNet')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--seed', type=int, default=9, metavar='S', help='random seed (default: 9)')
    parser.add_argument('--auto-continue', default=False, action='store_true', help='report frequency')
    parser.add_argument('--display-interval', type=int, default=10, help='per display-interval batches to' + ' display model training')
    parser.add_argument('--save-interval', type=int, default=10, help='per save-interval epochs to save model')

    parser.add_argument('--dataset-path', type=str, default='./dataset/', help='path to dataset')
    parser.add_argument('--train-dir', type=str, default='./imagenet/train', help='path to ImageNet training dataset')
    parser.add_argument('--val-dir', type=str, default='./imagenet/val', help='path to ImageNet validation dataset')

    parser.add_argument('--tunable-lif', default=False, action='store_true', help='use different learning rate for gating factors')
    parser.add_argument('--amp', default=False, action='store_true', help='use amp')
    parser.add_argument('--modeltag', type=str, default='SNN', help='decide the name of the experiment, this name will also be used as the checkpoint name')

    # configure the GLIF
    parser.add_argument('--gate', type=float, default=[0.6, 0.8, 0.6], nargs='+', help='initial gate')
    parser.add_argument('--static-gate', default=False, action='store_true', help='use static_gate')
    parser.add_argument('--static-param', default=False, action='store_true', help='use static_LIF_param')
    parser.add_argument('--channel-wise', default=False, action='store_true', help='use channel-wise')
    parser.add_argument('--softsimple', default=False, action='store_true', help='experiments on coarsely fused LIF')

    parser.add_argument('--soft-mode', default=False, action='store_true', help='use soft_gate')
    parser.add_argument('--t', type=int, default=3, help='the length of time window')
    parser.add_argument('--randomgate', default=False, action='store_true', help='activate uniform-randomly intialized gates')

    #define a dataset, default: cifar10
    parser.add_argument('--imagenet', default=False, action='store_true', help='experiments on ImageNet')
    parser.add_argument('--cifar100', default=False, action='store_true', help='experiments on cifar100')

    # define a model
    parser.add_argument('--stand18', default=False, action='store_true', help='use resnet18_stand')
    parser.add_argument('--cifarnet', default=False, action='store_true', help='use cifarnet')
    parser.add_argument('--MS18', default=False, action='store_true', help='experiments on ResNet-18MS')
    parser.add_argument('--MS34', default=False, action='store_true', help='experiments on ResNet-34MS')
    #ResNet-19 is the default option for CIFAR.
    #ResNet-34 is the default option for ImageNet.
    #To use any of the two models above, just clarify the task and DO NOT input any model commands. e.g., --stand18.

    args = parser.parse_args()

    return args




####################################################
# trainer & tester                                 #
#                                                  #
####################################################
def train(args, model, device, train_loader, optimizer, epoch, writer, criterion, scaler=None):
    t1 = time.time()
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()


def test(args, model, device, test_loader, epoch, writer, criterion, modeltag, dict_params, best= None):
    with torch.no_grad():
        acc = 0.
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda()
            outputs = F.softmax(model(inputs))
        
            #print(outputs)
            #print(targets)
        
            # 组织预测获得的值
            #pre[batch_idx*batch_size:(batch_idx+1)*batch_size] = outputs[:, :]
        
            labels = torch.zeros(batch_size, num_classes).scatter_(1, targets.view(-1, 1), 1)
            # 组织真实标签值
            #y_test[batch_idx*batch_size:(batch_idx+1)*batch_size] = labels[:, :]
        
            _, predicted = outputs.cpu().max(1)
        
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
        
        print('Testing acc:%.3f'%(100.*correct/total))

        

def seed_all(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def main():
    args = get_args()
    seed_all(args.seed)

    if torch.cuda.device_count() > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    epochs = 1#已经迭代的次数
    initial_dict = {'gate': [0.6, 0.8, 0.6], 'param': [tau, Vth, linear_decay, conduct],
                   't': steps, 'static_gate': True, 'static_param': False, 'time_wise': True, 'soft_mode': False}
    initial_dict['gate'] = args.gate
    initial_dict['static_gate'] = args.static_gate
    initial_dict['static_param'] = args.static_param
    initial_dict['time_wise'] = False
    initial_dict['soft_mode'] = args.soft_mode
    if args.t != steps:
        initial_dict['t']=args.t

    # In case time step is too large, we intuitively recommend to use the following code to alleviate the linear decay
    # initial_dict['param'][2] = initial_dict['param'][1]/(initial_dict['t'] * 2)


    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
    #---------------------------------------------------------------------------
    # 加载实验数据集
    transform = transforms.Compose(
        [transforms.Grayscale(),# 转成单通道的灰度图
         # 把值转成Tensor
        transforms.ToTensor()])

    dataset = ImageFolder("/kaggle/input/ddos-2019/Dataset-4/Dataset-4", 
                      transform=transform)

    # 切分，训练集和验证集
    random.seed(0)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split_point = int(0.8*len(indices))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    train_loader = DataLoader(dataset, batch_size=40,
                          sampler=SubsetRandomSampler(train_indices))
    val_loader = DataLoader(dataset, batch_size=40,
                         sampler=SubsetRandomSampler(test_indices))
    #---------------------------------------------------------------------------
    print('load data successfully')

    print(initial_dict)
    #prepare the model
    
    model = ResNet_18_stand_CW(lif_param=initial_dict, input_size=32, n_class=10)
           
    #model = ResNet_18_stand(lif_param=initial_dict, input_size=32, n_class=10)
        
    #model = CIFARNet(lif_param=initial_dict, input_size=32, n_class=10)
        

    if args.randomgate:
        randomize_gate(model)
        # model.randomize_gate
        print('randomized gate')

    modeltag = args.modeltag
    writer = SummaryWriter('./summaries/' + modeltag)
    print(model)
    dict_params = create_para_dict(args, model)
    # recording the initial GLIF parameters
    record_param(args, model, dict=dict_params, epoch=0, modeltag=modeltag)
    # classify GLIF-related params
    choice_param_name = ['alpha', 'beta', 'gamma']
    lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct', 'reVth']
    all_params = model.parameters()
    lif_params = []
    lif_choice_params = []
    lif_cal_params = []

    for pname, p in model.named_parameters():
        if pname.split('.')[-1] in choice_param_name:
            lif_params.append(p)
            lif_choice_params.append(p)
        elif pname.split('.')[-1] in lifcal_param_name:
            lif_params.append(p)
            lif_cal_params.append(p)
    # fetch id
    params_id = list(map(id, lif_params))
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    # optimizer & scheduler
    if args.tunable_lif:
        init_lr_diff = 10
        if args.imagenet:
            init_lr_diff = 1

        optimizer = torch.optim.SGD([
                {'params': other_params},
                {'params': lif_cal_params, "weight_decay": 0.},
                {'params': lif_choice_params, "weight_decay": 0., "lr":args.learning_rate / init_lr_diff}
            ],
                lr=args.learning_rate,
                momentum=0.9,
                weight_decay=args.weight_decay
            )
        scheduler = CosineAnnealingLR_Multi_Params_soft(optimizer,
                                                            T_max=[args.epochs, args.epochs, int(args.epochs)])
    else:
        optimizer = torch.optim.SGD([
            {'params': other_params},
            {'params': lif_params, "weight_decay": 0.}
        ],
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = Loss(args)
    device = torch.device("cuda" if use_gpu else "cpu")
    #Distributed computation
    if torch.cuda.is_available():
        loss_function = criterion.cuda()
    else:
        loss_function = criterion.cpu()

    if args.auto_continue:
        lastest_model = get_model(modeltag)
        if lastest_model is not None:
            checkpoint = torch.load(lastest_model, map_location='cpu')
            epochs = checkpoint['epoch']
            if torch.cuda.device_count() > 1:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                checkpoint = deletStrmodule(checkpoint)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('load from checkpoint, the epoch is {}'.format(epochs))
            dict_params = read_param(epoch=epochs, modeltag=modeltag)
            for i in range(epochs):
                scheduler.step()
            epochs += 1


    best = {'acc': 0., 'epoch': 0}

    if args.eval:
        lastest_model = get_model(modeltag, addr=args.eval_resume)
        if lastest_model is not None:
            epochs = -1
            checkpoint = torch.load(lastest_model, map_location='cpu')
            if args.imagenet:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                checkpoint = deletStrmodule(checkpoint)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            if torch.cuda.device_count() > 1:
                device = torch.device(local_rank)
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank],
                                                            output_device=local_rank,
                                                            find_unused_parameters=False)
            else:
                model = model.to(device)
            test(args, model, device, val_loader, epochs, writer, criterion=loss_function,
                 modeltag=modeltag, best=best, dict_params=dict_params)
        else:
            print('no model detected')
        exit(0)


    if torch.cuda.device_count() > 1:
        device = torch.device(local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank], output_device=local_rank,
                                                    find_unused_parameters=False)
    else:
        model = model.to(device)


    print('the random seed is {}'.format(args.seed))

    # amp
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    while (epochs <= args.epochs):
        train(args, model, device, train_loader, optimizer, epochs, writer, criterion=loss_function,
              scaler=scaler)
        if epochs % 5 == 0:
            torch.save(model, '/kaggle/working/model-'+str(epochs)+'.pt')
        else:
            pass
        test(args, model, device, val_loader, epochs, writer, criterion=loss_function,
                 modeltag=modeltag, best=best, dict_params=dict_params)
        scheduler.step()
        epochs += 1
    writer.close()
    torch.save(model, '/kaggle/working/model-last.pt')


if __name__ == "__main__":
    main()

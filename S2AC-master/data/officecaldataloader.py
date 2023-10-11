import torch
from torch.utils import data
from torchvision import datasets, transforms

Amazon_ImagePath = 'E:/pyworkspace/deep-coral-master/data/office_caltech_10/amazon'
Amazon_procee_path = 'E:/pyworkspace/deep-coral-master/data/office_caltech_10/amazonprocess'
dlsr_ImagePath = 'E:/pyworkspace/deep-coral-master/data/office_caltech_10/dslr'
dlsr_procee_path = 'E:/pyworkspace/deep-coral-master/data/office_caltech_10/dslrprocess'
webcam_ImagePath = 'E:/pyworkspace/deep-coral-master/data/office_caltech_10/webcam'
webcam_procee_path = 'E:/pyworkspace/deep-coral-master/data/office_caltech_10/webcamprocess'
caltech_ImagePath = 'E:/pyworkspace/deep-coral-master/data/office_caltech_10/caltech'
caltech_procee_path = 'E:/pyworkspace/deep-coral-master/data/office_caltech_10/caltechprocess'

def office_caltech_10_loader(name, batch_size):
    print('now load {} dataset......'.format(name))
    datas_path = {'Amazon':Amazon_ImagePath,
                  'Dlsr':dlsr_ImagePath,
                  'Webcam':webcam_ImagePath,
                  'Caltech':caltech_ImagePath}
    means = {'Amazon':[0.7526759863834752, 0.7602404336368335, 0.7672230537264952],
            'Dlsr':[0.4800972321610542, 0.45448788962546427, 0.3984687268544155],
             'Webcam':[0.6212397595947862, 0.6310887525595871, 0.61967591603597],
             'Caltech':[0.6355411211926015, 0.6248085544986164, 0.6234868050205421],
             'imagenet':[0.485,0.456,0.406]}
    stds = {'Amazon':[0.35313947664726364, 0.3452423321941799, 0.3401755908729863],
            'Dlsr':[0.21152113866843994, 0.20014552995087995, 0.2030288613620837],
             'Webcam':[0.2723014245125506, 0.2754929628529151, 0.27217746467659826],
            'Caltech': [0.27997598552932085, 0.2787646400947074, 0.27455120689616286],
            'imagenet':[0.229,0.224,0.406]}
    imgsize_W = 224
    imgsize_H = 224
    transform = [transforms.Resize((imgsize_W,imgsize_H)),
                 transforms.ToTensor(),
                 transforms.Normalize(means[name],stds[name])]
    dataset = datasets.ImageFolder(datas_path[name],transform = transforms.Compose(transform))
    # print(len(dataset))
    dataloader = data.DataLoader(
        dataset= dataset,
        batch_size=batch_size,
        shuffle = True,
        drop_last= True
    )
    return dataloader
import torch
from torch.utils import data
from torchvision import datasets, transforms

Art_ImagePath = 'E:/pyworkspace/deep-coral-master/data/OfficeHomeDataset_10072016/Art'
Art_procee_path = 'E:/pyworkspace/deep-coral-master/data/OfficeHomeDataset_10072016/Artprocess'
Clipart_ImagePath = 'E:/pyworkspace/deep-coral-master/data/OfficeHomeDataset_10072016/Clipart'
Clipart_procee_path = 'E:/pyworkspace/deep-coral-master/data/OfficeHomeDataset_10072016/Clipartprocess'
Product_ImagePath = 'E:/pyworkspace/deep-coral-master/data/OfficeHomeDataset_10072016/Product'
Product_procee_path = 'E:/pyworkspace/deep-coral-master/data/OfficeHomeDataset_10072016/Productprocess'
RealWorld_ImagePath = 'E:/pyworkspace/deep-coral-master/data/OfficeHomeDataset_10072016/Real World'
RealWorld_procee_path = 'E:/pyworkspace/deep-coral-master/data/OfficeHomeDataset_10072016/Real Worldprocess'

def officehomeloader(name, batch_size):
    print('now load {} dataset......'.format(name))
    datas_path = {'Art':Art_ImagePath,
                  'Clipart':Clipart_ImagePath,
                  'Product':Product_ImagePath,
                  'RealWorld':RealWorld_ImagePath}
    means = {'Art':[0.5155981174510142, 0.4827060958051683, 0.4454448953797395],
            'Clipart':[0.5983396151830842, 0.5769766079725827, 0.5513267560970686],
             'Product':[0.7472607019196044, 0.7348014121309954, 0.726138245874433],
             'RealWorld':[0.6067654113395327, 0.5755390924559117, 0.5424632404888994],
             'imagenet':[0.485,0.456,0.406]}
    stds = {'Art':[0.23892518669692242, 0.23470071584851987, 0.22829736754617477],
            'Clipart':[0.296632159857041, 0.2948531819872861, 0.2995061623826004],
             'Product':[0.27248417841672135, 0.27872645107290317, 0.2837751301550555],
            'RealWorld': [0.23997337607757055, 0.24067685999585856, 0.24572975056479768],
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
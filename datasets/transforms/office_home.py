from torchvision import transforms
from .randaugment import RandAugmentMC
from .autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy


def office_home_train(mean, std):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform


def office_home_test(mean, std):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform


class OHMultiView(object):
    '''
        Multi-view transformation for office home
    '''
    def __init__(self, mean, std, views='ww', aug='auto'):
        assert all(v in 'wstqk' for v in views)  # qk for strong transform for query and key
        assert aug in ['rand', 'auto']
        self.views = views
        print('view:', views)
        self.weak = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.strong_q = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.strong_k = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=3, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.strong = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10) if aug == 'rand' else ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        ret_views = []
        for v in self.views:
            if v == 'w':
                ret_views.append(self.weak(x))
            elif v == 's':
                ret_views.append(self.strong(x))
            elif v == 't':
                ret_views.append(self.test(x))
            elif v == 'q':
                ret_views.append(self.strong_q(x))
            elif v == 'k':
                ret_views.append(self.strong_k(x))
        if len(ret_views) > 1:
            return ret_views
        else:
            return ret_views[0]

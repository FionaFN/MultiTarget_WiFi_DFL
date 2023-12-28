import torch
import torch.nn as nn
import torch.nn.functional as F
from yacs.config import CfgNode as CN
from .swin_transformer import SwinTransformer
from .cbam import CBAM


def cfg_swin(
    img_size=40,
    C_in=1,
    patchsize=5, 
    windowsize=4,
    H=60,
    W=60,
):
    _C = CN()
    ### Data settings
    _C.DATA = CN()
    _C.DATA.IMG_SIZE = img_size
    ### Model settings
    _C.MODEL = CN()
    _C.MODEL.SWIN = CN()
    _C.MODEL.SWIN.IN_CHANS = C_in

    _C.MODEL.SWIN.PATCH_SIZE = patchsize
    _C.MODEL.SWIN.WINDOW_SIZE = windowsize #1,2,4,8

    _C.MODEL.SWIN.EMBED_DIM = 96
    _C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2] 
    _C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]

    _C.MODEL.SWIN.MLP_RATIO = 4.
    _C.MODEL.SWIN.QKV_BIAS = True
    _C.MODEL.SWIN.QK_SCALE = None
    _C.MODEL.SWIN.APE = False
    _C.MODEL.SWIN.PATCH_NORM = True # Number of classes, overwritten in data preparation
    _C.MODEL.NUM_CLASSES = H*W
    # Dropout rate
    _C.MODEL.DROP_RATE = 0.0
    # Drop path rate
    _C.MODEL.DROP_PATH_RATE = 0.1
    ### Training settings
    _C.TRAIN = CN()
    _C.TRAIN.USE_CHECKPOINT = False
    ### for acceleration
    _C.FUSED_WINDOW_PROCESS = False
    #
    config = _C.clone()
    return config

def load_swin(config):
    model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                            patch_size=config.MODEL.SWIN.PATCH_SIZE,
                            in_chans=config.MODEL.SWIN.IN_CHANS,
                            num_classes=config.MODEL.NUM_CLASSES,
                            embed_dim=config.MODEL.SWIN.EMBED_DIM,
                            depths=config.MODEL.SWIN.DEPTHS,
                            num_heads=config.MODEL.SWIN.NUM_HEADS,
                            window_size=config.MODEL.SWIN.WINDOW_SIZE,
                            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                            qk_scale=config.MODEL.SWIN.QK_SCALE,
                            drop_rate=config.MODEL.DROP_RATE,
                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                            ape=config.MODEL.SWIN.APE,
                            patch_norm=config.MODEL.SWIN.PATCH_NORM,
                            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                            fused_window_process=config.FUSED_WINDOW_PROCESS)
    return model

class mCBAM(nn.Module):
    def __init__(self, 
                 C: int = 1,# input channels of each feature
                 r: int = 1, #channel reduction ratio
                 t: int = 1, #times of repetition
                 initialfuse: str = 'add', # 'cat'
                 pool = nn.MaxPool3d((2,1,1),stride=(2,1,1)), #nn.AdaptiveAvgPool3d((1, None, None)),
                ):
        super(mCBAM, self).__init__()
        self.initialfuse = initialfuse
        
        if self.initialfuse=='add': 
            C_inblock = C
        elif self.initialfuse=='cat': 
            C_inblock = 2*C

        self.cbam = CBAM(C_inblock,r)
        self.repeat = t
        self.pool = pool # pooling along channel dimension
        
    def forward(self, x, y):
        #initial feature integration
        if self.initialfuse=='add': 
            z = x+y 
        elif self.initialfuse=='cat': 
            z = torch.cat([x,y], 1)
            
        for t in range(self.repeat):
            z = self.cbam(z)
        
        if self.initialfuse=='cat': 
            z = self.pool(z) # pooling along channel dimension
        
        return z

class Model(nn.Module):
    def __init__(self,
                 key: str,
                ):
        super(Model, self).__init__()
        self.swin = load_swin(cfg_swin())
        if key=='mCBAMadd1': self.fuse = mCBAM(C=1, r=1, t=1, initialfuse='add')
        if key=='mCBAMadd2': self.fuse = mCBAM(C=1, r=1, t=2, initialfuse='add')

    def forward(self, dRSS, prior, numGrids=60):
        pre_pred = self.swin(dRSS)
        pre_pred = pre_pred.reshape(-1,1,numGrids,numGrids)
        fused = self.fuse(pre_pred, prior)
        return pre_pred, fused

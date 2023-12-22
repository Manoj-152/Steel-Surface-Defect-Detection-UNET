import torch.nn as nn
import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def contract_block(self, in_channels, mid_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size),
                    nn.LeakyReLU(),
                    nn.BatchNorm2d(mid_channels),
                    nn.Conv2d(mid_channels, out_channels, kernel_size),
                    nn.LeakyReLU(),
                    nn.BatchNorm2d(out_channels)
                )
        return block

    def expand_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = nn.Sequential(
                    nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
                    nn.LeakyReLU(),
                    nn.BatchNorm2d(mid_channel),
                    nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
                    nn.LeakyReLU(),
                    nn.BatchNorm2d(mid_channel),
                    nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                    )
        return  block
        
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3, padding=1):
        block = nn.Sequential(
                    nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=padding),
                    nn.LeakyReLU(),
                    nn.BatchNorm2d(mid_channel),
                    nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=padding),
                    nn.LeakyReLU(),
                    nn.BatchNorm2d(mid_channel),
                    nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=padding),
                    nn.LeakyReLU(),
                    nn.BatchNorm2d(out_channels),
                )
        return  block
    
    def __init__(self, in_channel, out_channel):
        super(Model, self).__init__()
        #Encode
        self.conv_encode1 = self.contract_block(in_channels=in_channel, mid_channels=64, out_channels=64)
        self.conv_encode2 = self.contract_block(64, 128, 256)
        self.conv_encode3 = self.contract_block(256, 256, 512)
        self.conv_maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
                            torch.nn.Conv2d(kernel_size=3, in_channels=512, out_channels=256, padding=1),
                            torch.nn.LeakyReLU(),
                            torch.nn.BatchNorm2d(256),
                            torch.nn.Conv2d(kernel_size=3, in_channels=256, out_channels=256, padding=1),
                            torch.nn.LeakyReLU(),
                            torch.nn.BatchNorm2d(256),
                            torch.nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
                            )
        # Decode
        self.conv_decode3 = self.expand_block(1024, 512, 256)
        self.conv_decode2 = self.expand_block(512, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)
        
    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_maxpool1 = self.conv_maxpool(encode_block1)
        encode_block2 = self.conv_encode2(encode_maxpool1)
        encode_maxpool2 = self.conv_maxpool(encode_block2)
        encode_block3 = self.conv_encode3(encode_maxpool2)
        encode_maxpool3 = self.conv_maxpool(encode_block3)
               
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_maxpool3)
        bottleneck1 = F.interpolate(bottleneck1, size=(25,193), mode='bicubic', align_corners=False)
        
        #Decode
        decode_block3 = self.conv_decode3(torch.cat([encode_block3,bottleneck1], 1))
        decode_block3 = F.interpolate(decode_block3, size=(58,394), mode='bicubic', align_corners=False)
        decode_block2 = self.conv_decode2(torch.cat([encode_block2,decode_block3], 1))
        decode_block2 = F.interpolate(decode_block2, size=(124,796), mode='bicubic', align_corners=False)
        final_layer = self.final_layer(torch.cat([encode_block1,decode_block2], 1))
        final_layer = F.interpolate(final_layer, size=(128,800), mode='bicubic', align_corners=False)
        final_layer = F.softmax(final_layer,dim=1)
        
        return final_layer

if __name__ == "__main__":
    model = Model(3,5).cuda()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)
    inp = torch.randn(8, 3, 128, 800).cuda()
    out = model(inp)
    print(out.shape)

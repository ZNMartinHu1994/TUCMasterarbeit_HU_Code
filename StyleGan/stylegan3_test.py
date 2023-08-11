import torch, os
import pickle
import cv2 as cv
import numpy as np
import torch.nn.functional as F
import time

print(r'pytorch version:', torch.__version__)

'''---------------------------The following is the definition of the parameter section---------------------------------'''
#the tuning parameter only needs to be changed here, but not anywhere else

gen_img_num = 5000  # How many images are generated
dsize = (56, 379)  # (width, height) The size to which the generated image will be scaled
weight_path = r'C:/work/models/3.StyleGAN/1.bad-50+20/00000-stylegan3-t-good-gpus1-batch2-gamma12.2/network-snapshot-000020.pkl'  # The path where the trained weights are located

'''---------------------------Above is the definition parameter section------------------------------------------------'''

with open(weight_path, 'rb') as f:
    G = pickle.load(f)['G_ema'].eval().cuda()  # torch.nn.Module
# Save Image
os.makedirs('result_image', exist_ok=True)
print('Begin generating images...')
imgs = []
t_list = []
for i in range(gen_img_num):
    z = torch.randn([1, G.z_dim]).cuda()  # latent codes
    t = time.time()
    img = G(z, G.c_dim)  # Generate images
    t_list.append(time.time() - t)  # recording time
    # imgs.append(img)
    # imgs = torch.cat(imgs, dim=0)
    img = np.clip(img.detach().cpu().numpy(), -1, 1).transpose([0, 2, 3, 1])  # (b,h,w,c)
    img = np.flip(img, axis=-1).squeeze(0)  # (h,w,c)
    if dsize is not None:  # Scaling output images
        img = cv.resize(img, dsize=(dsize[1], dsize[0]))
    img = (img * 127.5 + 127.5).astype('uint8')

    # for i, img in enumerate(imgs):
    save_path = 'result_image/%s.bmp' % (i + 1)
    cv.imwrite(save_path, img)
print('Has generated %s of images to :%s，time-consuming:%s' % (gen_img_num, os.path.join(os.getcwd(), 'result_image'), np.sum(t_list[1:])))

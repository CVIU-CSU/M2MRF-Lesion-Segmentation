import os

import PIL.Image as image

root = '../data'
datasets = ['IDRID', 'DDR']


def do_augment(image_name, mask_name):
    img = image.open(os.path.join(img_path, image_name))
    mask = image.open(os.path.join(mask_path, mask_name))

    img.save(os.path.join(img_path, image_name[:-4] + ".jpg"))
    img.transpose(image.ROTATE_180).save(os.path.join(img_path, image_name[:-4] + "_180.jpg"))
    img.transpose(image.ROTATE_90).save(os.path.join(img_path, image_name[:-4] + "_90.jpg"))
    img.transpose(image.ROTATE_270).save(os.path.join(img_path, image_name[:-4] + "_270.jpg"))
    img.transpose(image.FLIP_LEFT_RIGHT).save(os.path.join(img_path, image_name[:-4] + "_horizontal.jpg"))
    img.transpose(image.FLIP_TOP_BOTTOM).save(os.path.join(img_path, image_name[:-4] + "_vertical.jpg"))

    mask.save(os.path.join(mask_path, mask_name[:-4] + ".png"))
    mask.transpose(image.ROTATE_180).save(os.path.join(mask_path, mask_name[:-4] + "_180.png"))
    mask.transpose(image.ROTATE_90).save(os.path.join(mask_path, mask_name[:-4] + "_90.png"))
    mask.transpose(image.ROTATE_270).save(os.path.join(mask_path, mask_name[:-4] + "_270.png"))
    mask.transpose(image.FLIP_LEFT_RIGHT).save(os.path.join(mask_path, mask_name[:-4] + "_horizontal.png"))
    mask.transpose(image.FLIP_TOP_BOTTOM).save(os.path.join(mask_path, mask_name[:-4] + "_vertical.png"))


if __name__ == "__main__":

    for dataset in datasets:
        print(dataset)
        img_path = os.path.join(root, dataset, 'image/train')
        mask_path = os.path.join(root, dataset, 'label/train/annotations')

        for mask_name in os.listdir(mask_path):
            image_name = mask_name[:-4] + '.jpg'
            if not os.path.exists(os.path.join(img_path, image_name)):
                print('not found ' + os.path.join(mask_path, mask_name))
                continue
            print(mask_name)
            do_augment(image_name, mask_name)

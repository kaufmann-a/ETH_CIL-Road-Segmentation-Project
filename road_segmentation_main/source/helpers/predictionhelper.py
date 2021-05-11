import os
import torch
import cv2

from source.postprocessing.postprocessing import postprocess


def patch_to_label(foreground_threshold, patch):
    # assign a label to a patch
    df = torch.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image, image_nr, patch_size=16, foreground_threshold=0.25):
    # iterate over prediction, just use every 16th pixel
    for j in range(0, image.shape[1], patch_size):
        for i in range(0, image.shape[0], patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(foreground_threshold, patch)
            yield ("{:03d}_{}_{},{}".format(image_nr, j, i, label))


# TODO: Handle if morphological operations aren't defined.
def runpostprocessing(preds_list, folder, postprocessingparams, image_number_list, patch_size, foreground_threshold):
    folder_postprocessed = os.path.join(folder, "pred-masks-postprocessed")
    if not os.path.exists(folder_postprocessed):
        os.makedirs(folder_postprocessed)

    with open(os.path.join(folder, 'postprocessed_submission.csv'), 'w') as f:
        f.write('id,prediction\n')
        image_nr_list_idx = 0

        for i in range(len(preds_list)):
            preds = preds_list[i]
            preds = preds.cpu().numpy()
            preds = preds.astype('uint8')
            postprocessed_img = postprocess(preds, postprocessingparams.morphology)
            cv2.imwrite(f"{folder_postprocessed}/pred_{image_number_list[i]}.png", 255 * postprocessed_img)
            postprocessed_img = torch.tensor(postprocessed_img)
            postprocessed_img = postprocessed_img.to(torch.double)
            f.writelines('{}\n'.format(s)
                         for s in mask_to_submission_strings(image=postprocessed_img,
                                                             patch_size=patch_size,
                                                             image_nr=image_number_list[image_nr_list_idx],
                                                             foreground_threshold=foreground_threshold))
            image_nr_list_idx += 1
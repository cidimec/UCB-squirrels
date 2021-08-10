from pickle import load
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

print(os.getcwd())

# model = load(open('model.pkl', 'rb'))
# scaler = load(open('scaler.pkl', 'rb'))
# pca = load(open('pca.pkl', 'rb'))

model = load(open('OAKD8S_model.pkl', 'rb'))
scaler = load(open('OAKD8S_scaler.pkl', 'rb'))
pca = load(open('OAKD8S_pca.pkl', 'rb'))

def GEI_generator(sil_file, size=64, debug=False):
    stack_GEI = []
    lenfiles = len(sil_file)
    if debug:
        plt.figure(figsize=(20, int(lenfiles // 10)))

    for idimg, img in enumerate(sil_file):
        if debug:
            plt.subplot((lenfiles //15)+ 1, 15, idimg+ 1)

        # Silhouette extraction
        contours1, _ = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours1, -1, 255, -1)

        if (len(contours1)>0):
            ncoun= np.concatenate(contours1)[:, 0, :]
            x1, y1 = np.min(ncoun, axis=0)
            x2, y2 = np.max(ncoun, axis=0)
            silhouette = img[y1:y2, x1:x2]

            # Normalizae silhouette
            factor = size/max(silhouette.shape)
            height = round(factor*silhouette.shape[0])
            width = round(factor*silhouette.shape[1])
            if(height>width):
                nor_sil = cv2.resize(silhouette,(width,height))
                #         print(nor_sil.shape)
                # We add a background of the shape size x size
                portion_body = 0.3                                                      # We take the upper part of the body to center the image and avoid the legs
                moments = cv2.moments(nor_sil[0:int(nor_sil.shape[0]*portion_body),])
                w = round(moments['m10']/moments['m00'])
                background = np.zeros((size, size))
                shift = round((size/2)-w)
                #         print('center:',w,' shift:',shift)
                if(shift<0 or shift+nor_sil.shape[1]>size):
                    shift = round((size-nor_sil.shape[1])/2)
                background[:,shift:nor_sil.shape[1]+shift] = nor_sil

                stack_GEI.append(background)
                if debug:
                    plt.xticks([])
                    plt.yticks([])
                    plt.imshow(background,'gray')
                    plt.title(str(idimg))
                    # plt.subplots_adjust(wspace=0.05, hspace=0.01)
    if stack_GEI == []:
        GEI = np.zeros((size, size))
        print('\tNo Files Found')
    else:
        GEI = np.mean(np.array(stack_GEI), axis=0)

    # GEI[int(size*0.12):int(size*0.68),:]=0
    return GEI, stack_GEI


def inference(GEI):
    flatten = GEI.flatten()[np.newaxis]
    scaled = scaler.transform(flatten)
    norm = pca.transform(scaled)
    return model.predict(norm)

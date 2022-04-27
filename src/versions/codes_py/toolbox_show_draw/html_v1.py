# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Borrowed and developed from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/html.py
# This tool does not care about functional / non-functional
from ..np_ext.np_image_io_v1 import gifWrite
from ..py_ext.misc_v1 import mkdir_full
import dominate
from dominate.tags import *
import os
import skimage.io as io
import numpy as np
import shutil


class HTML:
    def __init__(self, web_dir, title, reflesh=0):

        if os.path.isdir(web_dir):
            shutil.rmtree(web_dir)
        mkdir_full(web_dir + 'images/')

        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_header_h4(self, str):
        with self.doc:
            h4(str)

    def add_header_h5(self, str):
        with self.doc:
            h5(str)

    def add_text(self, s):
        with self.doc:
            p(s)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=400):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(im):  # zhuzhu
                                br()
                                img(style="width:%dpx" % width, src=link)
                            br()
                            if not(txt is None):
                                p(txt)

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


def storeImagesAndReturnLinks(str_store, str_link, strs_imgs, imgsContents):
    links = []
    assert len(strs_imgs) == len(imgsContents)
    for i in range(len(strs_imgs)):
        if len(imgsContents[i].shape) in [2, 3]:  # It is png
            if len(imgsContents[i].shape) == 2:
                assert imgsContents[i].shape[0] > 5
                assert imgsContents[i].shape[1] > 5
            if 'float32' in str(imgsContents[i].dtype):
                tmp = imgsContents[i].copy()
                tmp[tmp < 0] = 0.
                tmp[tmp > 1] = 1.
                tmp = (tmp * 255.).astype(np.uint8)
                io.imsave(str_store + strs_imgs[i] + '.png', tmp)
            else:
                io.imsave(str_store + strs_imgs[i] + '.png', imgsContents[i])
            links.append(str_link + strs_imgs[i] + '.png')
        elif len(imgsContents[i].shape) == 4:  # It is gif
            gifWrite(str_store + strs_imgs[i] + '.gif', imgsContents[i])
            links.append(str_link + strs_imgs[i] + '.gif')
        else:
            raise ValueError('Shape of imgsContents[i] is wierd %s' % str(imgsContents[i].shape))

    return links


class HTMLStepperAbstract(object):
    def __init__(self, logDir, singleHtmlSteps, htmlName):  # consider to use htmlName to represent different experiments.
        # Static member
        assert os.path.isdir(logDir)
        if not logDir.endswith('/'):
            logDir += '/'
        self.logDir = logDir
        self.htmlName = htmlName
        self.singleHtmlSteps = singleHtmlSteps

        # dynamics
        self.htmlStepCount = float('inf')
        self.currentFileID = None
        self.html = None

    def _printLossTotal(self, loss):
        raise NotImplementedError

    def _printLossEach(self, loss, j):
        raise NotImplementedError

    def step2(self, summary0=None, txt0=None, brInds=(0, ), headerMessage=None, subMessage=None):
        assert brInds[0] == 0
        if txt0 is not None:
            assert len(summary0) == len(txt0)
        if self.htmlStepCount >= self.singleHtmlSteps:  # reset
            assert os.path.isdir(self.logDir)
            mkdir_full(self.logDir + 'html_' + self.htmlName + '/')
            tmp = [int(x[5:]) for x in os.listdir(self.logDir + 'html_' + self.htmlName + '/') if x.startswith('html_')]
            if tmp:  # safe if  # tmp is a non-empty set
                maxHtmlFileID = max(tmp)
            else:  # tmp is an empty set
                maxHtmlFileID = -1
            self.currentFileID = maxHtmlFileID + 1
            self.html = HTML(self.logDir + 'html_' + self.htmlName + '/' + 'html_%06d/' % self.currentFileID,
                             self.htmlName)  # zhuzhu
            self.htmlStepCount = 0

        if headerMessage is not None:
            self.html.add_header(headerMessage)
        if subMessage is not None:
            self.html.add_header_h4(subMessage)
        if summary0 is not None:
            links = storeImagesAndReturnLinks(
                self.logDir + 'html_' + self.htmlName + '/html_%06d/images/%s_%06d_%06d_' %
                    (self.currentFileID, self.htmlName, self.currentFileID, self.htmlStepCount),
                'images/%s_%06d_%06d_' %
                    (self.htmlName, self.currentFileID, self.htmlStepCount),
                list(summary0.keys()),
                list(summary0.values()),
            )
            for o in range(len(brInds)):
                head = brInds[o]
                if head >= len(summary0):
                    break
                if o == len(brInds) - 1 or brInds[o + 1] >= len(summary0):
                    tail = len(summary0)
                else:
                    tail = brInds[o + 1]
                if txt0 is None:
                    self.html.add_images(
                        list(summary0.keys())[head:tail],
                        [None for _ in range(head, tail)], links[head:tail])
                else:
                    self.html.add_images(
                        list(summary0.keys())[head:tail],
                        [txt0[i] for i in range(head, tail)], links[head:tail])
        self.html.save()
        self.htmlStepCount += 1

    def step(self, iterCount, losses, visual, txts=None, brInds=(0, )):  # functional, but file system changed  # losses can be OrderedDict as in here, or also a TreeLog
        assert brInds[0] == 0
        if txts is not None:
            assert len(visual.summary) == len(txts[0])
        if self.htmlStepCount >= self.singleHtmlSteps:  # reset
            assert os.path.isdir(self.logDir)
            mkdir_full(self.logDir + 'html_' + self.htmlName + '/')
            tmp = [int(x[5:]) for x in os.listdir(self.logDir + 'html_' + self.htmlName + '/') if x.startswith('html_')]
            if tmp:  # safe if  # tmp is a non-empty set
                maxHtmlFileID = max(tmp)
            else:  # tmp is an empty set
                maxHtmlFileID = -1
            self.currentFileID = maxHtmlFileID + 1
            self.html = HTML(self.logDir + 'html_' + self.htmlName + '/' + 'html_%06d/' % self.currentFileID, self.htmlName)  # zhuzhu
            self.htmlStepCount = 0

        self.html.add_header('Iter %d %s' % (iterCount, self.htmlName))  # zhuzhu
        self._printLossTotal(losses)

        for j in range(visual.m):
            links = storeImagesAndReturnLinks(
                self.logDir + 'html_' + self.htmlName + '/html_%06d/images/%s_%08d_%06d_%02d_%02d_%08d_%08d_' % \
                    (self.currentFileID, self.htmlName, iterCount, self.currentFileID, j, visual.datasetID[j], visual.index[j], visual.insideBatchIndex[j]),  # zhuzhu
                'images/%s_%08d_%06d_%02d_%02d_%08d_%08d_' % (self.htmlName, iterCount, self.currentFileID, j, visual.datasetID[j], visual.index[j], visual.insideBatchIndex[j]),  # zhuzhu
                list(visual.summary.keys()),
                [x[j] for x in visual.summary.values()],
            )
            self.html.add_header('Iter = %d, Dataset = %s, SampleID = %d, InsideBatchID %d' % \
                                 (iterCount, visual.dataset[j], visual.index[j], visual.insideBatchIndex[j]))
            self._printLossEach(losses, visual.insideBatchIndex[j])
            for o in range(len(brInds)):
                head = brInds[o]
                if head >= len(visual.summary):
                    break
                if o == len(brInds) - 1 or brInds[o + 1] >= len(visual.summary):
                    tail = len(visual.summary)
                else:
                    tail = brInds[o + 1]
                if txts is None:
                    self.html.add_images(list(visual.summary.keys())[head:tail], [None for _ in range(head, tail)], links[head:tail])
                else:
                    self.html.add_images(list(visual.summary.keys())[head:tail], [txts[j][i] for i in range(head, tail)], links[head:tail])
            # self.html.add_images(visual.summary.keys(), [None for _ in visual.summary.keys()], links)
        self.html.save()
        self.htmlStepCount += 1


class HTMLStepper(HTMLStepperAbstract):
    pass

# Do more class extension directly in your code!

class HTMLStepperODTotal(HTMLStepperAbstract):
    # Override
    def _printLossTotal(self, loss_od):
        st = ''
        for k in loss_od.keys():
            assert type(loss_od[k]) is float
            st += ' %s: %.5f,' % (k, loss_od[k])
        self.html.add_header_h4(st)

    # Override
    def _printLossEach(self, loss_od, j):
        pass


class HTMLStepperNoPrinting(HTMLStepperAbstract):
    # Override
    def _printLossTotal(self, loss_none):
        pass

    # Override
    def _printLossEach(self, loss_none, j):
        pass

class HTMLStepperTreeLog(HTMLStepperAbstract):  # dedicated to our framework
    # Notes for __init__(logDir, singleHtmlSteps, htmlName): htmlName in here should be something like 'G_%s_%s_%s_%s' % (P, D, S, R)

    # Override
    def _printLossTotal(self, treeLog):
        for t in treeLog.keys():
            treeLog[t].print_stack(2, self.html.add_header_h4)

    # Override
    def _printLossEach(self, treeLog, j):
        for t in treeLog.keys():
            treeLog[t].print_stack(4, self.html.add_text, j)


if __name__ == '__main__':
    html = HTML('./', 'test_html')
    html.add_header('hello world')

    ims = []
    txts = []
    links = []
    for n in range(1, 5):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        # links.append('image_%d.png' % n)  # zhuzhu
        links.append('/home/zhusz/link/play/html/image_%d.png' % n)
    html.add_images(ims, txts, links)

    ims = []
    txts = []
    links = []
    for n in range(1, 5):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        # links.append('image_%d.png' % n)  # zhuzhu
        links.append('/home/zhusz/link/play/html/image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()

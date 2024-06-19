"""
refer.py

This is a lightweight refactor of the original RefCOCO/RefCOCO+/RefCOCOg data loading code for Python3; all rights
are retained by the original author Licheng Yu; this code abides by the license in the original repository found
here: https://github.com/lichengunc/refer/

Note that instead of using the `external/mask` code, we add a python dependency: `pycocotools`

This interface provides access to four datasets:
1) refclef
2) refcoco
3) refcoco+
4) refcocog
split by unc and google

The following API functions are defined:
REFER      - REFER api class
getRefIds  - get ref ids that satisfy given filter conditions.
getAnnIds  - get ann ids that satisfy given filter conditions.
getImgIds  - get image ids that satisfy given filter conditions.
getCatIds  - get category ids that satisfy given filter conditions.
loadRefs   - load refs with the specified ref ids.
loadAnns   - load anns with the specified ann ids.
loadImgs   - load images with the specified image ids.
loadCats   - load category names with the specified category ids.
getRefBox  - get ref's bounding box [x, y, w, h] given the ref_id
showRef    - show image, segmentation or box of the referred object with the ref
getMask    - get mask and area of the referred object given ref
showMask   - show mask of the referred object given ref
"""
__author__ = "licheng"

import itertools
import json
import os.path as osp
import pickle
import time
from pprint import pprint
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from pycocotools import mask


# ruff: noqa: E721
class REFER:
    def __init__(self, data_root: str, dataset: str = "refcoco", splitBy: str = "unc") -> None:
        """
        Provide `data_root` folder which contains `refclef`, `refcoco`, `refcoco+`, and `refcocog`.
        Additionally provide `dataset` name and `splitBy` information.
        """
        print(f"Loading dataset {dataset} into memory...")
        self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
        self.DATA_DIR = osp.join(data_root, dataset)
        if dataset in ["refcoco", "refcoco+", "refcocog"]:
            self.IMAGE_DIR = osp.join(data_root, "train2014")
        elif dataset == "refclef":
            self.IMAGE_DIR = osp.join(data_root, "saiapr_tc-12")
        else:
            print(f"No refer dataset is called [{dataset}]")
            exit(1)

        # Load refs from data/dataset/refs(dataset).json
        tic = time.time()
        ref_file = osp.join(self.DATA_DIR, "refs(" + splitBy + ").p")
        self.data = {"dataset": dataset, "refs": pickle.load(open(ref_file, "rb"))}

        # Load annotations from data/dataset/instances.json
        instances_file = osp.join(self.DATA_DIR, "instances.json")
        instances = json.load(open(instances_file, "r"))
        self.data["images"] = instances["images"]
        self.data["annotations"] = instances["annotations"]
        self.data["categories"] = instances["categories"]

        # Create index
        self.createIndex()
        print(f"DONE (t={time.time() - tic:.2f}s)")

    def createIndex(self) -> None:
        """
        Create mappings:
            1)  Refs: 	 	    {ref_id: ref}
            2)  Anns: 	 	    {ann_id: ann}
            3)  Imgs:	        {image_id: image}
            4)  Cats: 	 	    {category_id: category_name}
            5)  Sents:     	    {sent_id: sent}
            6)  imgToRefs: 	    {image_id: refs}
            7)  imgToAnns: 	    {image_id: anns}
            8)  refToAnn:  	    {ref_id: ann}
            9)  annToRef:  	    {ann_id: ref}
            10) catToRefs: 	    {category_id: refs}
            11) sentToRef: 	    {sent_id: ref}
            12) sentToTokens:   {sent_id: tokens}
        """
        print("Creating index....")

        # Fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data["annotations"]:
            Anns[ann["id"]] = ann
            imgToAnns[ann["image_id"]] = [*imgToAnns.get(ann["image_id"], []), ann]

        for img in self.data["images"]:
            Imgs[img["id"]] = img

        for cat in self.data["categories"]:
            Cats[cat["id"]] = cat["name"]

        # Fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data["refs"]:
            # IDs
            ref_id = ref["ref_id"]
            ann_id = ref["ann_id"]
            category_id = ref["category_id"]
            image_id = ref["image_id"]

            # Add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = [*imgToRefs.get(image_id, []), ref]
            catToRefs[category_id] = [*catToRefs.get(category_id, []), ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # Add mapping of sent
            for sent in ref["sentences"]:
                Sents[sent["sent_id"]] = sent
                sentToRef[sent["sent_id"]] = ref
                sentToTokens[sent["sent_id"]] = sent["tokens"]

        # Create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens

        print("Index created.")

    # ruff: noqa: C901
    def getRefIds(self, image_ids=None, cat_ids=None, ref_ids=None, split="") -> List[int]:
        if ref_ids is None:
            ref_ids = []
        if cat_ids is None:
            cat_ids = []
        if image_ids is None:
            image_ids = []
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data["refs"]
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data["refs"]

            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref["category_id"] in cat_ids]

            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref["ref_id"] in ref_ids]

            if not len(split) == 0:
                if split in ["testA", "testB", "testC"]:
                    # We also consider testAB, testBC, ...
                    refs = [ref for ref in refs if split[-1] in ref["split"]]
                elif split in ["testAB", "testBC", "testAC"]:
                    # Rarely used I guess...
                    refs = [ref for ref in refs if ref["split"] == split]
                elif split == "test":
                    refs = [ref for ref in refs if "test" in ref["split"]]
                elif split == "train" or split == "val":
                    refs = [ref for ref in refs if ref["split"] == split]
                else:
                    print(f"No such split [{split}]")
                    exit(1)

        ref_ids = [ref["ref_id"] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=None, cat_ids=None, ref_ids=None) -> List[str]:
        if ref_ids is None:
            ref_ids = []
        if cat_ids is None:
            cat_ids = []
        if image_ids is None:
            image_ids = []
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann["id"] for ann in self.data["annotations"]]
        else:
            if not len(image_ids) == 0:
                # List of [anns]
                lists = [self.imgToAnns[image_id] for image_id in image_ids if image_id in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data["annotations"]

            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann["category_id"] in cat_ids]

            ann_ids = [ann["id"] for ann in anns]
            if not len(ref_ids) == 0:
                set(ann_ids).intersection(set([self.Refs[ref_id]["ann_id"] for ref_id in ref_ids]))

        return ann_ids

    def getImgIds(self, ref_ids=None) -> List[str]:
        if ref_ids is None:
            ref_ids = []
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]
        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]["image_id"] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()

        return image_ids

    def getCatIds(self) -> List[str]:
        return list(self.Cats.keys())

    def loadRefs(self, ref_ids=None) -> List[str]:
        if ref_ids is None:
            ref_ids = []
        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]

        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=None) -> List[str]:
        if ann_ids is None:
            ann_ids = []
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int or type(ann_ids) == str:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=None) -> List[str]:
        if image_ids is None:
            image_ids = []
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=None) -> List[str]:
        if cat_ids is None:
            cat_ids = []
        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif type(cat_ids) == int:
            return [self.Cats[cat_ids]]

    def getRefBox(self, ref_id) -> List[int]:
        ann = self.refToAnn[ref_id]
        return ann["bbox"]  # [x, y, w, h]

    def showRef(self, ref: str, seg_box: str = "seg") -> None:
        # Show image
        ax = plt.gca()
        image = self.Imgs[ref["image_id"]]
        img = io.imread(osp.join(self.IMAGE_DIR, image["file_name"]))
        ax.imshow(img)

        # Show refer expression
        for sid, sent in enumerate(ref["sentences"]):
            print(f"{sid + 1}. {sent['sent']}")

        # Show segmentations
        if seg_box == "seg":
            ann_id = ref["ann_id"]
            ann = self.Anns[ann_id]
            polygons = []
            color = []
            c = "none"
            if type(ann["segmentation"][0]) == list:
                # Polygon used for refcoco*
                for seg in ann["segmentation"]:
                    poly = np.array(seg).reshape((len(seg) / 2, 2))
                    polygons.append(Polygon(poly, True, alpha=0.4))
                    color.append(c)

                # Thick yellow polygon
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 1, 0, 0), linewidths=3, alpha=1)
                ax.add_collection(p)

                # Thin red polygon
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 0, 0, 0), linewidths=1, alpha=1)
                ax.add_collection(p)

            else:
                # Mask used for refclef
                rle = ann["segmentation"]
                m = mask.decode(rle)
                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = np.array([2.0, 166.0, 101.0]) / 255
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((img, m * 0.5)))

        # Show bounding-box
        elif seg_box == "box":
            ann_id = ref["ann_id"]
            ann = self.Anns[ann_id]
            bbox = self.getRefBox(ref["ref_id"])
            box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor="green", linewidth=3)
            ax.add_patch(box_plot)

    def getMask(self, ref: str) -> np.ndarray:
        # Return mask, area and mask-center
        ann = self.refToAnn[ref["ref_id"]]
        image = self.Imgs[ref["image_id"]]

        # Polygon
        if type(ann["segmentation"][0]) == list:
            rle = mask.frPyObjects(ann["segmentation"], image["height"], image["width"])
        else:
            rle = ann["segmentation"]

        # Sometimes there are multiple binary map (corresponding to multiple segs)
        m = mask.decode(rle)
        m = np.sum(m, axis=2)
        m = m.astype(np.uint8)

        # Compute area --> should be close to ann['area']
        area = sum(mask.area(rle))
        return {"mask": m, "area": area}

    def showMask(self, ref: str) -> None:
        M = self.getMask(ref)
        msk = M["mask"]
        ax = plt.gca()
        ax.imshow(msk)


if __name__ == "__main__":
    refer = REFER(".", dataset="refcocog", splitBy="google")
    ref_identifiers = refer.getRefIds()
    print(len(ref_identifiers))
    print(len(refer.Imgs))
    print(len(refer.imgToRefs))

    ref_identifiers = refer.getRefIds(split="train")
    print(f"There are {len(ref_identifiers)} training referred objects.")
    for ref_identifier in ref_identifiers:
        ref_instance = refer.loadRefs(ref_identifier)[0]
        if len(ref_instance["sentences"]) < 2:
            continue

        pprint(ref_instance)
        print(f"The label is {refer.Cats[ref_instance['category_id']]}.")
        plt.figure()
        refer.showRef(ref_instance, seg_box="box")
        plt.show()

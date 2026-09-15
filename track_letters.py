"""Conservative cyan-glyph tracking for this fixed, authored Scene11 layout.

This is a classical-vision estimate, not simulator ground truth. Occluded or
poorly matched letters are omitted. Do not use after randomizing scene colors.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

ANCHORS = {'O': (457, 282), 'P': (518, 282), 'Q': (577, 282),
           'A': (637, 283), 'S': (694, 282), 'U': (795, 284),
           'E': (517, 325), 'F': (579, 325), 'D': (638, 325),
           'K': (693, 325), 'M': (750, 325), 'H': (511, 373),
           'I': (577, 373), 'J': (638, 373), 'L': (695, 373), 'C': (752, 371)}


def mask(image):
    b, g, r = cv2.split(image)
    return ((b > 130) & (g > 100) & (r < 100)).astype(np.uint8) * 255


class Tracker:
    def __init__(self, reference, geometry, targets='ACE', camera_forward_shift=0.):
        self.targets = targets
        self.component_diagnostics = {}
        self.anchors = dict(ANCHORS)
        self.geometry = json.loads(Path(geometry).read_text())
        ref = mask(cv2.imread(str(reference)))
        count, labels, stats, centers = cv2.connectedComponentsWithStats(ref)
        self.templates = {}
        self.extra_templates = []
        centers_by_letter = {}
        for letter, point in ANCHORS.items():
            candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 15]
            idx = min(candidates, key=lambda i: np.linalg.norm(centers[i] - point))
            x, y, w, h, _ = stats[idx]
            self.templates[letter] = ref[y-3:y+h+3, x-3:x+w+3]
            centers_by_letter[letter] = [x+w/2, y+h/2]
        if camera_forward_shift:
            letters = list(centers_by_letter)
            h, _ = cv2.findHomography(np.float32([centers_by_letter[l] for l in letters]),
                np.float32([self.geometry['letter_cube_'+l]['position'][:2] for l in letters]), cv2.RANSAC, .004)
            inv = np.linalg.inv(h)
            for letter in letters:
                x,y = self.geometry['letter_cube_'+letter]['position'][:2]
                pixel = inv @ [x,y-camera_forward_shift,1.]
                self.anchors[letter] = tuple(map(int, np.rint(pixel[:2]/pixel[2])))

    def add_visual_exemplar(self, letter, image, pixel):
        """Add a visually confirmed appearance; never substitutes object position."""
        binary = mask(cv2.imread(str(image)))
        count, _, stats, centers = cv2.connectedComponentsWithStats(binary)
        candidates = [i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=20]
        idx = min(candidates,key=lambda i: np.linalg.norm(centers[i]-pixel))
        if np.linalg.norm(centers[idx]-pixel)>10:
            raise ValueError('No glyph near annotated exemplar')
        x,y,w,h,_ = stats[idx]
        self.extra_templates.append((letter,binary[y-3:y+h+3,x-3:x+w+3]))

    def component_match(self, binary, letter):
        """Handle perspective scale and modest yaw on moved, isolated glyphs."""
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        reference = self.templates[letter][3:-3,3:-3]
        rh,rw = reference.shape
        candidates = []
        for idx in range(1,count):
            x,y,w,h,area = stats[idx]
            pixel = [x+w/2,y+h/2]
            if area<20 or h<8 or max(w,h)>80 or not .4<(w/h)/(rw/rh)<2.5:
                continue
            if np.max(np.abs(np.asarray(pixel)-self.anchors[letter]))>240:
                continue
            glyph = (labels[y:y+h,x:x+w]==idx).astype(np.uint8)*255
            pad=max(w,h)
            glyph=cv2.copyMakeBorder(glyph,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=0)
            center=((glyph.shape[1]-1)/2,(glyph.shape[0]-1)/2)
            score=-1.
            for angle in range(-30,31,5):
                matrix=cv2.getRotationMatrix2D(center,angle,1.)
                rotated=cv2.warpAffine(glyph,matrix,(glyph.shape[1],glyph.shape[0]),flags=cv2.INTER_NEAREST)
                pts=cv2.findNonZero(rotated)
                bx,by,bw,bh=cv2.boundingRect(pts)
                normalized=cv2.resize(rotated[by:by+bh,bx:bx+bw],(rw,rh),interpolation=cv2.INTER_NEAREST)
                value=float(cv2.matchTemplate(normalized,reference,cv2.TM_CCOEFF_NORMED)[0,0])
                score=max(score,value)
            candidates.append({'score':score,'pixel':pixel,'match_method':'component_rotation_normalization'})
        candidates.sort(key=lambda item:item['score'],reverse=True)
        self.component_diagnostics[letter] = candidates[:5]
        if not candidates:
            return None
        best=candidates[0]
        best['distinct_match_margin']=best['score']-(candidates[1]['score'] if len(candidates)>1 else -1.)
        return best if best['score']>=.8 and best['distinct_match_margin']>=.10 else None

    def locate(self, image, identified=None):
        binary = mask(cv2.imread(str(image)))
        found = {}
        for letter, template in list(self.templates.items()) + self.extra_templates:
            u, v = self.anchors[letter]
            # Reference letters should remain near their authored pixels.
            radius = 240 if letter in self.targets else 24
            x0, x1 = max(0, u-radius), min(binary.shape[1], u+radius)
            y0, y1 = max(0, v-radius), min(binary.shape[0], v+radius)
            search = binary[y0:y1, x0:x1]
            best = None
            candidates = []
            for scale in np.linspace(.8, 1.85, 22):
                t = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                if t.shape[0] > search.shape[0] or t.shape[1] > search.shape[1]:
                    continue
                scores = cv2.matchTemplate(search, t, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(scores)
                pixel = [x0+loc[0]+t.shape[1]/2, y0+loc[1]+t.shape[0]/2]
                candidates.append((float(score), pixel))
                scores[max(0,loc[1]-25):loc[1]+26, max(0,loc[0]-25):loc[0]+26] = -1
                _, other_score, _, other_loc = cv2.minMaxLoc(scores)
                candidates.append((float(other_score), [x0+other_loc[0]+t.shape[1]/2, y0+other_loc[1]+t.shape[0]/2]))
                if best is None or score > best['score']:
                    best = {'score': float(score), 'pixel': [x0+loc[0]+t.shape[1]/2, y0+loc[1]+t.shape[0]/2]}
            if best:
                other = max((score for score, pixel in candidates
                             if np.linalg.norm(np.asarray(pixel)-best['pixel']) > 25), default=-1.)
                best['distinct_match_margin'] = best['score'] - other
            if best and best['score'] >= (.84 if letter in self.targets else .77) and (letter not in self.targets or best['distinct_match_margin'] >= .06):
                if letter not in found or best['score'] > found[letter]['score']:
                    found[letter] = best
        for letter in self.targets:
            if identified is None and letter not in found and letter in self.templates:
                candidate=self.component_match(binary,letter)
                if candidate:
                    found[letter]=candidate
        if identified is not None:
            # API supplies identity; measured cyan components supply the pixel
            # center. Do not silently fall back to old identity templates.
            count, _, stats, centers=cv2.connectedComponentsWithStats(binary)
            used=set()
            for letter in self.targets:
                found.pop(letter,None)
                if letter not in identified:continue
                candidates=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=20 and max(stats[i,2:4])<=80]
                if not candidates:continue
                idx=min(candidates,key=lambda i: np.linalg.norm(centers[i]-identified[letter]))
                distance=float(np.linalg.norm(centers[idx]-identified[letter]))
                if distance>18 or idx in used:continue
                used.add(idx)
                x,y,w,h,_=stats[idx]
                found[letter]={'pixel':[float(x+w/2),float(y+h/2)],'identity_source':'astra_image_decision',
                               'pixel_source':'cyan_connected_component','api_pixel_snap_distance':distance}
        refs = [l for l in found if l not in self.targets]
        if len(refs) < 6:
            return {'ok': False, 'reason': 'Too few visible reference letters', 'matches': found}
        pixels = np.float32([found[l]['pixel'] for l in refs])
        world = np.float32([self.geometry['letter_cube_'+l]['position'][:2] for l in refs])
        homography, inliers = cv2.findHomography(pixels, world, cv2.RANSAC, .004)
        if homography is None or int(inliers.sum()) < 6:
            return {'ok': False, 'reason': 'Reference-plane fit failed', 'matches': found}
        estimates = {}
        for letter in self.targets:
            if letter not in found:
                continue
            p = homography @ [*found[letter]['pixel'], 1.]
            estimates[letter] = {**found[letter], 'estimated_xyz': [float(p[0]/p[2]), float(p[1]/p[2]), .777]}
        return {'ok': True, 'source': 'api_identified_cyan_glyphs' if identified is not None else 'cyan_glyph_template_tracking', 'frame': str(image),
                'reference_inliers': int(inliers.sum()), 'letters': estimates,
                'limitations': 'Approximate XY for upright blocks on table, several-mm uncertainty. Missing letters may be occluded; not a success signal.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', type=Path)
    p.add_argument('--reference', default='runs/initial.jpg')
    p.add_argument('--geometry', default='runs/scene_authored_geometry.json')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = Tracker(args.reference, args.geometry).locate(args.image)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + '\n')
    print(text)

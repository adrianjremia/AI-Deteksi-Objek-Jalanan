# YOLOv5 🚀 oleh Ultralytics, lisensi GPL-3.0
"""
Memvalidasi akurasi model YOLOv5 yang telah dilatih pada dataset khusus

Penggunaan:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse  # Mengimpor modul argparse untuk menangani argumen baris perintah
import json  # Mengimpor modul json untuk manipulasi file JSON
import os  # Mengimpor modul os untuk operasi sistem
import sys  # Mengimpor modul sys untuk memanipulasi sistem Python
from pathlib import Path  # Mengimpor Path dari modul pathlib untuk manipulasi path
from threading import Thread  # Mengimpor Thread dari modul threading untuk operasi multithreading

import numpy as np  # Mengimpor pustaka NumPy untuk operasi numerik
import torch  # Mengimpor pustaka PyTorch untuk machine learning
from tqdm import tqdm  # Mengimpor pustaka tqdm untuk progress bar

FILE = Path(__file__).absolute()  # Mendapatkan path absolut dari file skrip ini
sys.path.append(FILE.parents[0].as_posix())  # Menambahkan direktori induk dari file ini ke path

from models.experimental import attempt_load  # Mengimpor fungsi attempt_load dari models.experimental
from utils.datasets import create_dataloader  # Mengimpor fungsi create_dataloader dari utils.datasets
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr  # Mengimpor fungsi utilitas dari utils.general
from utils.metrics import ap_per_class, ConfusionMatrix  # Mengimpor ap_per_class dan ConfusionMatrix dari utils.metrics
from utils.plots import plot_images, output_to_target, plot_study_txt  # Mengimpor fungsi plotting dari utils.plots
from utils.torch_utils import select_device, time_sync  # Mengimpor fungsi utilitas dari utils.torch_utils
from utils.callbacks import Callbacks  # Mengimpor Callbacks dari utils.callbacks

def save_one_txt(predn, save_conf, shape, file):
    # Menyimpan satu hasil txt
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalisasi gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # xywh dinormalisasi
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # format label
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')  # menulis ke file

def save_one_json(predn, jdict, path, class_map):
    # Menyimpan satu hasil JSON {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem  # mendapatkan ID gambar
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center ke sudut kiri atas
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})  # menambahkan hasil ke jdict

def process_batch(detections, labels, iouv):
    """
    Mengembalikan matriks prediksi yang benar. Kedua set kotak dalam format (x1, y1, x2, y2).
    Argumen:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Mengembalikan:
        correct (Array[N, 10]), untuk 10 level IoU
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)  # inisialisasi tensor correct
    iou = box_iou(labels[:, 1:], detections[:, :4])  # menghitung IoU
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU di atas ambang dan kelas cocok
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, deteksi, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv  # menentukan prediksi yang benar
    return correct

@torch.no_grad()  # Dekorator untuk menonaktifkan gradient
def run(data,
        weights=None,  # path model.pt
        batch_size=32,  # ukuran batch
        imgsz=640,  # ukuran inferensi (piksel)
        conf_thres=0.001,  # ambang batas kepercayaan
        iou_thres=0.6,  # ambang batas IoU NMS
        task='val',  # train, val, test, speed atau study
        device='',  # perangkat cuda, misalnya 0 atau 0,1,2,3 atau cpu
        single_cls=False,  # perlakukan sebagai dataset kelas tunggal
        augment=False,  # inferensi yang ditingkatkan
        verbose=False,  # output verbose
        save_txt=False,  # simpan hasil ke *.txt
        save_hybrid=False,  # simpan hasil hibrida label+prediksi ke *.txt
        save_conf=False,  # simpan kepercayaan dalam label --save-txt
        save_json=False,  # simpan file hasil COCO-JSON
        project='runs/val',  # simpan ke project/name
        name='exp',  # simpan ke project/name
        exist_ok=False,  # proyek/nama yang ada ok, tidak menambah
        half=True,  # gunakan inferensi FP16 half-precision
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        ):
    # Inisialisasi/muat model dan atur perangkat
    training = model is not None  # apakah dipanggil oleh train.py
    if training:  # dipanggil oleh train.py
        device = next(model.parameters()).device  # mendapatkan perangkat model
    else:  # dipanggil langsung
        device = select_device(device, batch_size=batch_size)  # memilih perangkat

        # Direktori
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # menambah run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # membuat direktori

        # Memuat model
        model = attempt_load(weights, map_location=device)  # memuat model FP32
        gs = max(int(model.stride.max()), 32)  # ukuran grid (stride maksimum)
        imgsz = check_img_size(imgsz, s=gs)  # memeriksa ukuran gambar

    # Half precision
    half &= device.type != 'cpu'  # half precision hanya didukung pada CUDA
    if half:
        model.half()  # mengubah model ke FP16

    # Konfigurasi
    model.eval()  # menetapkan mode evaluasi
    is_coco = type(data['val']) is str and data['val'].endswith('coco/val2017.txt')  # dataset COCO
    nc = 1 if single_cls else int(data['nc'])  # jumlah kelas
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # vektor iou untuk mAP@0.5:0.95
    niou = iouv.numel()  # jumlah iou

    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # menjalankan sekali
        task = task if task in ('train', 'val', 'test') else 'val'  # path ke gambar train/val/test
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]  # membuat dataloader

    seen = 0  # jumlah gambar yang dilihat
    confusion_matrix = ConfusionMatrix(nc=nc)  # inisialisasi matriks kebingungan
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}  # nama kelas
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))  # peta kelas
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')  # format string untuk output
    p, r, f1, mp, mr, map50, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  # inisialisasi metrik
    loss = torch.zeros(3, device=device)  # inisialisasi loss
    jdict, stats, ap, ap_class = [], [], [], []  # inisialisasi daftar

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):  # iterasi melalui batch
        t_ = time_sync()  # mencatat waktu awal
        img = img.to(device, non_blocking=True)  # mengirim gambar ke perangkat
        img = img.half() if half else img.float()  # mengubah tipe data gambar
        img /= 255.0  # mengubah rentang nilai gambar
        targets = targets.to(device)  # mengirim target ke perangkat
        nb, _, height, width = img.shape  # mendapatkan ukuran batch, channel, tinggi, lebar
        t = time_sync()  # mencatat waktu setelah preprocessing
        t0 += t - t_

        # Menjalankan model
        out, train_out = model(img, augment=augment)  # mendapatkan output inferensi dan pelatihan
        t1 += time_sync() - t  # mencatat waktu inferensi

        # Menghitung loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # menghitung box, obj, cls loss

        # Menjalankan NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # mengubah target ke piksel
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # untuk autolabelling
        t = time_sync()  # mencatat waktu sebelum NMS
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)  # menjalankan NMS
        t2 += time_sync() - t  # mencatat waktu setelah NMS

        # Statistik per gambar
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)  # jumlah label
            tcls = labels[:, 0].tolist() if nl else []  # kelas target
            path, shape = Path(paths[si]), shapes[si][0]  # path dan shape gambar
            seen += 1  # meningkatkan jumlah gambar yang dilihat

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))  # jika tidak ada prediksi
                continue

            # Prediksi
            if single_cls:
                pred[:, 5] = 0  # menetapkan semua kelas ke 0
            predn = pred.clone()  # klon prediksi
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # mengubah skala kotak prediksi

            # Evaluasi
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # kotak target
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # mengubah skala kotak target
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # label native-space
                correct = process_batch(predn, labelsn, iouv)  # memproses batch
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)  # memperbarui matriks kebingungan
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)  # tidak ada label yang benar
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # menambahkan statistik

            # Menyimpan/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))  # menyimpan ke txt
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # menambahkan ke jdict
            callbacks.on_val_image_end(pred, predn, path, names, img[si])  # memanggil callback

        # Plot gambar
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()  # memplot gambar
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # prediksi
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()  # memplot gambar

    # Menghitung statistik
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # menggabungkan statistik
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)  # menghitung AP per kelas
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()  # menghitung metrik rata-rata
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # jumlah target per kelas
    else:
        nt = torch.zeros(1)  # tidak ada target

    # Mencetak hasil
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # format cetak
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))  # mencetak hasil

    # Mencetak hasil per kelas
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))  # mencetak hasil per kelas

    # Mencetak kecepatan
    t = tuple(x / seen * 1E3 for x in (t0, t1, t2))  # kecepatan per gambar
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)  # bentuk gambar
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)  # mencetak kecepatan

    # Plot
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))  # memplot matriks kebingungan
        callbacks.on_val_end()  # memanggil callback

    # Menyimpan JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # menentukan weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # path file json anotasi
        pred_json = str(save_dir / f"{w}_predictions.json")  # path file json prediksi
        print(f'\nEvaluating pycocotools mAP... saving {pred_json}...')  # mencetak pesan evaluasi
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)  # menyimpan hasil ke file json

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])  # memeriksa persyaratan pycocotools
            from pycocotools.coco import COCO  # mengimpor COCO dari pycocotools
            from pycocotools.cocoeval import COCOeval  # mengimpor COCOeval dari pycocotools

            anno = COCO(anno_json)  # inisialisasi API anotasi
            pred = anno.loadRes(pred_json)  # inisialisasi API prediksi
            eval = COCOeval(anno, pred, 'bbox')  # inisialisasi evaluasi
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # ID gambar yang dievaluasi
            eval.evaluate()  # mengevaluasi
            eval.accumulate()  # mengakumulasi hasil
            eval.summarize()  # merangkum hasil
            map, map50 = eval.stats[:2]  # memperbarui hasil (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')  # menangani pengecualian

    # Mengembalikan hasil
    model.float()  # untuk pelatihan
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''  # jumlah label yang disimpan
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")  # mencetak informasi penyimpanan hasil
    maps = np.zeros(nc) + map  # inisialisasi maps
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]  # memperbarui maps
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t  # mengembalikan metrik dan waktu

def parse_opt():
    parser = argparse.ArgumentParser(prog='val.py')  # membuat parser untuk argumen baris perintah
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='path dataset.yaml')  # argumen path dataset
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='path model.pt')  # argumen path model
    parser.add_argument('--batch-size', type=int, default=32, help='ukuran batch')  # argumen ukuran batch
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='ukuran inferensi (piksel)')  # argumen ukuran gambar
    parser.add_argument('--conf-thres', type=float, default=0.001, help='ambang batas kepercayaan')  # argumen ambang batas kepercayaan
    parser.add_argument('--iou-thres', type=float, default=0.6, help='ambang batas IoU NMS')  # argumen ambang batas IoU
    parser.add_argument('--task', default='val', help='train, val, test, speed atau study')  # argumen tugas
    parser.add_argument('--device', default='', help='perangkat cuda, misalnya 0 atau 0,1,2,3 atau cpu')  # argumen perangkat
    parser.add_argument('--single-cls', action='store_true', help='perlakukan sebagai dataset kelas tunggal')  # argumen kelas tunggal
    parser.add_argument('--augment', action='store_true', help='inferensi yang ditingkatkan')  # argumen inferensi yang ditingkatkan
    parser.add_argument('--verbose', action='store_true', help='laporkan mAP per kelas')  # argumen laporan mAP per kelas
    parser.add_argument('--save-txt', action='store_true', help='simpan hasil ke *.txt')  # argumen simpan hasil ke txt
    parser.add_argument('--save-hybrid', action='store_true', help='simpan hasil hibrida label+prediksi ke *.txt')  # argumen simpan hasil hibrida
    parser.add_argument('--save-conf', action='store_true', help='simpan kepercayaan dalam label --save-txt')  # argumen simpan kepercayaan
    parser.add_argument('--save-json', action='store_true', help='simpan file hasil COCO-JSON')  # argumen simpan file JSON
    parser.add_argument('--project', default='runs/val', help='simpan ke project/name')  # argumen nama proyek
    parser.add_argument('--name', default='exp', help='simpan ke project/name')  # argumen nama hasil
    parser.add_argument('--exist-ok', action='store_true', help='proyek/nama yang ada ok, tidak menambah')  # argumen proyek/nama yang ada ok
    parser.add_argument('--half', action='store_true', help='gunakan inferensi FP16 half-precision')  # argumen gunakan inferensi FP16
    opt = parser.parse_args()  # mengurai argumen
    opt.save_json |= opt.data.endswith('coco.yaml')  # mengatur save_json jika data berakhir dengan coco.yaml
    opt.save_txt |= opt.save_hybrid  # mengatur save_txt jika save_hybrid diaktifkan
    opt.data = check_file(opt.data)  # memeriksa file
    return opt  # mengembalikan argumen yang diurai

def main(opt):
    set_logging()  # mengatur logging
    print(colorstr('val: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))  # mencetak argumen yang diurai
    check_requirements(requirements=FILE.parent / 'requirements.txt', exclude=('tensorboard', 'thop'))  # memeriksa persyaratan

    if opt.task in ('train', 'val', 'test'):  # menjalankan secara normal
        run(**vars(opt))  # menjalankan fungsi run dengan argumen yang diurai
    elif opt.task == 'speed':  # uji kecepatan
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                save_json=False, plots=False)  # menjalankan uji kecepatan
    elif opt.task == 'study':  # menjalankan berbagai pengaturan dan menyimpan/memplot
        # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # sumbu x (ukuran gambar)
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # nama file untuk disimpan
            y = []  # sumbu y
            for i in x:  # ukuran gambar
                print(f'\nRunning {f} point {i}...')  # mencetak pesan menjalankan
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, save_json=opt.save_json, plots=False)  # menjalankan dan mendapatkan hasil
                y.append(r + t)  # menambahkan hasil dan waktu ke y
            np.savetxt(f, y, fmt='%10.4g')  # menyimpan hasil ke file txt
        os.system('zip -r study.zip study_*.txt')  # mengompres file txt menjadi zip
        plot_study_txt(x=x)  # memplot hasil

if __name__ == "__main__":
    opt = parse_opt()  # mengurai argumen
    main(opt)  # menjalankan fungsi utama dengan argumen yang diurai

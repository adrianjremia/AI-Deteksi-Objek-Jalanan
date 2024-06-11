# YOLOv5 oleh Ultralytics, lisensi GPL-3.0
"""
Menjalankan inferensi pada gambar, video, direktori, stream, dll.

Penggunaan:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse  # Mengimpor modul argparse untuk menangani argumen baris perintah
import sys  # Mengimpor modul sys untuk memanipulasi sistem Python
import time  # Mengimpor modul time untuk mengukur waktu
from pathlib import Path  # Mengimpor Path dari modul pathlib untuk manipulasi path

import cv2  # Mengimpor pustaka OpenCV untuk pemrosesan gambar dan video
import numpy as np  # Mengimpor pustaka NumPy untuk operasi numerik
import torch  # Mengimpor pustaka PyTorch untuk machine learning
import torch.backends.cudnn as cudnn  # Mengimpor backend CUDA untuk PyTorch

FILE = Path(__file__).absolute()  # Mendapatkan path absolut dari file skrip ini
sys.path.append(FILE.parents[0].as_posix())  # Menambahkan direktori induk dari file ini ke path

from models.experimental import attempt_load  # Mengimpor fungsi attempt_load dari models.experimental
from utils.datasets import LoadStreams, LoadImages  # Mengimpor fungsi LoadStreams dan LoadImages dari utils.datasets
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, is_ascii, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box  # Mengimpor fungsi utilitas dari utils.general
from utils.plots import Annotator, colors  # Mengimpor Annotator dan colors dari utils.plots
from utils.torch_utils import select_device, load_classifier, time_sync  # Mengimpor fungsi utilitas dari utils.torch_utils

@torch.no_grad()  # Dekorator untuk menonaktifkan gradient
def run(weights='yolov5s.pt',  # path model.pt
        source='data/images',  # file/dir/URL/glob, 0 untuk webcam
        imgsz=640,  # ukuran inferensi (piksel)
        conf_thres=0.25,  # ambang batas kepercayaan
        iou_thres=0.45,  # ambang batas IOU NMS
        max_det=1000,  # deteksi maksimum per gambar
        device='',  # perangkat cuda, misalnya 0 atau 0,1,2,3 atau cpu
        view_img=False,  # tampilkan hasil
        save_txt=False,  # simpan hasil ke *.txt
        save_conf=False,  # simpan kepercayaan dalam label --save-txt
        save_crop=False,  # simpan kotak prediksi yang dipotong
        nosave=False,  # tidak menyimpan gambar/video
        classes=None,  # filter berdasarkan kelas: --class 0, atau --class 0 2 3
        agnostic_nms=False,  # NMS agnostik kelas
        augment=False,  # inferensi yang ditingkatkan
        visualize=False,  # visualisasikan fitur
        update=False,  # perbarui semua model
        project='runs/detect',  # simpan hasil ke project/name
        name='exp',  # simpan hasil ke project/name
        exist_ok=False,  # proyek/nama yang ada ok, tidak menambah
        line_thickness=3,  # ketebalan kotak batas (piksel)
        hide_labels=False,  # sembunyikan label
        hide_conf=False,  # sembunyikan kepercayaan
        half=False,  # gunakan inferensi FP16 half-precision
        ):
    save_img = not nosave and not source.endswith('.txt')  # menyimpan gambar inferensi
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))  # menentukan apakah sumber adalah webcam

    # Direktori
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # menambah run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # membuat direktori

    # Inisialisasi
    set_logging()  # mengatur logging
    device = select_device(device)  # memilih perangkat
    half &= device.type != 'cpu'  # half precision hanya didukung pada CUDA

    # Memuat model
    w = weights[0] if isinstance(weights, list) else weights  # menentukan weights
    classify, suffix = False, Path(w).suffix.lower()  # menentukan suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
    stride, names = 64, [f'class{i}' for i in range(1000)]  # menetapkan default
    if pt:
        model = attempt_load(weights, map_location=device)  # memuat model FP32
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # mendapatkan nama kelas
        if half:
            model.half()  # mengubah ke FP16
        if classify:  # classifier tahap kedua
            modelc = load_classifier(name='resnet50', n=2)  # inisialisasi
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))  # memeriksa persyaratan
        import onnxruntime  # mengimpor onnxruntime
        session = onnxruntime.InferenceSession(w, None)  # membuat sesi inferensi ONNX
    else:  # model TensorFlow
        check_requirements(('tensorflow>=2.4.1',))  # memeriksa persyaratan
        import tensorflow as tf  # mengimpor TensorFlow
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # impor yang dibungkus
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()  # membuat definisi graf
            graph_def.ParseFromString(open(w, 'rb').read())  # parsing dari file
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")  # membungkus fungsi graf beku
        elif saved_model:
            model = tf.keras.models.load_model(w)  # memuat model yang disimpan
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # memuat model TFLite
            interpreter.allocate_tensors()  # mengalokasikan tensor
            input_details = interpreter.get_input_details()  # mendapatkan detail input
            output_details = interpreter.get_output_details()  # mendapatkan detail output
            int8 = input_details[0]['dtype'] == np.uint8  # apakah model TFLite dikuantisasi uint8
    imgsz = check_img_size(imgsz, s=stride)  # memeriksa ukuran gambar
    ascii = is_ascii(names)  # nama-nama adalah ascii (gunakan PIL untuk UTF-8)

    # Dataloader
    if webcam:
        view_img = check_imshow()  # memeriksa apakah imshow tersedia
        cudnn.benchmark = True  # menetapkan True untuk mempercepat inferensi ukuran gambar konstan
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)  # memuat stream
        bs = len(dataset)  # ukuran batch
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)  # memuat gambar
        bs = 1  # ukuran batch
    vid_path, vid_writer = [None] * bs, [None] * bs  # inisialisasi variabel

    # Menjalankan inferensi
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # menjalankan sekali
    t0 = time.time()  # mencatat waktu awal
    for path, img, im0s, vid_cap in dataset:
        if onnx:
            img = img.astype('float32')  # mengubah tipe data gambar
        else:
            img = torch.from_numpy(img).to(device)  # mengonversi gambar ke tensor PyTorch
            img = img.half() if half else img.float()  # mengubah tipe data gambar
        img = img / 255.0  # mengubah rentang nilai gambar
        if len(img.shape) == 3:
            img = img[None]  # menambahkan dimensi batch

        # Inferensi
        t1 = time_sync()  # mencatat waktu inferensi
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False  # menentukan path untuk visualisasi
            pred = model(img, augment=augment, visualize=visualize)[0]  # mendapatkan prediksi
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))  # mendapatkan prediksi ONNX
        else:  # model tensorflow (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # mengubah gambar ke numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()  # mendapatkan prediksi tensorflow pb
            elif saved_model:
                pred = model(imn, training=False).numpy()  # mendapatkan prediksi tensorflow saved_model
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)  # menetapkan tensor input
                interpreter.invoke()  # menjalankan model
                pred = interpreter.get_tensor(output_details[0]['index'])  # mendapatkan tensor output
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)  # mengubah prediksi ke tensor PyTorch

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  # menjalankan NMS
        t2 = time_sync()  # mencatat waktu setelah NMS

        # Classifier tahap kedua (opsional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)  # menerapkan classifier

        # Memproses prediksi
        for i, det in enumerate(pred):  # deteksi per gambar
            if webcam:  # ukuran batch >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count  # mendapatkan informasi gambar
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)  # mendapatkan informasi gambar

            p = Path(p)  # mengubah path ke objek Path
            save_path = str(save_dir / p.name)  # menentukan path penyimpanan
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # menentukan path file txt
            s += '%gx%g ' % img.shape[2:]  # menambahkan informasi ukuran gambar
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalisasi gain whwh
            imc = im0.copy() if save_crop else im0  # untuk save_crop
            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)  # membuat annotator
            if len(det):
                # Mengubah skala kotak dari img_size ke ukuran im0
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()  # mengubah skala kotak

                # Mencetak hasil
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # deteksi per kelas
                    s += f"{n} {names[int(c)]}{' ' * (n > 1)}, "  # menambahkan ke string

                # Menulis hasil
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Menulis ke file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # xywh dinormalisasi
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # format label
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')  # menulis ke file

                    if save_img or save_crop or view_img:  # Menambahkan kotak batas ke gambar
                        c = int(cls)  # kelas integer
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # label
                        annotator.box_label(xyxy, label, color=colors(c, True))  # menambahkan label ke kotak
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)  # menyimpan kotak yang dipotong

            # Mencetak waktu (inferensi + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')  # mencetak informasi waktu

            # Streaming hasil
            im0 = annotator.result()  # mendapatkan hasil annotator
            if view_img:  # jika ingin menampilkan gambar
                cv2.imshow(str(p), im0)  # menampilkan gambar
                cv2.waitKey(1)  # 1 milidetik

            # Menyimpan hasil (gambar dengan deteksi)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)  # menyimpan gambar
                else:  # 'video' atau 'stream'
                    if vid_path[i] != save_path:  # video baru
                        vid_path[i] = save_path  # menetapkan path video
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # melepaskan penulis video sebelumnya
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)  # mendapatkan FPS
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # mendapatkan lebar frame
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # mendapatkan tinggi frame
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]  # menetapkan default
                            save_path += '.mp4'  # menambahkan ekstensi .mp4
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))  # membuat penulis video
                    vid_writer[i].write(im0)  # menulis frame ke video

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''  # jumlah label yang disimpan
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")  # mencetak informasi penyimpanan hasil

    if update:
        strip_optimizer(weights)  # memperbarui model (untuk memperbaiki SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')  # mencetak waktu total

def parse_opt():
    parser = argparse.ArgumentParser()  # membuat parser untuk argumen baris perintah
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='path model.pt')  # argumen path model
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 untuk webcam')  # argumen sumber data
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='ukuran inferensi h,w')  # argumen ukuran gambar
    parser.add_argument('--conf-thres', type=float, default=0.25, help='ambang batas kepercayaan')  # argumen ambang batas kepercayaan
    parser.add_argument('--iou-thres', type=float, default=0.45, help='ambang batas IoU NMS')  # argumen ambang batas IoU
    parser.add_argument('--max-det', type=int, default=1000, help='deteksi maksimum per gambar')  # argumen deteksi maksimum
    parser.add_argument('--device', default='', help='perangkat cuda, misalnya 0 atau 0,1,2,3 atau cpu')  # argumen perangkat
    parser.add_argument('--view-img', action='store_true', help='tampilkan hasil')  # argumen tampilkan gambar
    parser.add_argument('--save-txt', action='store_true', help='simpan hasil ke *.txt')  # argumen simpan hasil ke txt
    parser.add_argument('--save-conf', action='store_true', help='simpan kepercayaan dalam label --save-txt')  # argumen simpan kepercayaan
    parser.add_argument('--save-crop', action='store_true', help='simpan kotak prediksi yang dipotong')  # argumen simpan kotak prediksi yang dipotong
    parser.add_argument('--nosave', action='store_true', help='jangan menyimpan gambar/video')  # argumen jangan menyimpan gambar/video
    parser.add_argument('--classes', nargs='+', type=int, help='filter berdasarkan kelas: --class 0, atau --class 0 2 3')  # argumen filter kelas
    parser.add_argument('--agnostic-nms', action='store_true', help='NMS agnostik kelas')  # argumen NMS agnostik kelas
    parser.add_argument('--augment', action='store_true', help='inferensi yang ditingkatkan')  # argumen inferensi yang ditingkatkan
    parser.add_argument('--visualize', action='store_true', help='visualisasikan fitur')  # argumen visualisasikan fitur
    parser.add_argument('--update', action='store_true', help='perbarui semua model')  # argumen perbarui semua model
    parser.add_argument('--project', default='runs/detect', help='simpan hasil ke project/name')  # argumen nama proyek
    parser.add_argument('--name', default='exp', help='simpan hasil ke project/name')  # argumen nama hasil
    parser.add_argument('--exist-ok', action='store_true', help='proyek/nama yang ada ok, tidak menambah')  # argumen proyek/nama yang ada ok
    parser.add_argument('--line-thickness', default=3, type=int, help='ketebalan kotak batas (piksel)')  # argumen ketebalan kotak batas
    parser.add_argument('--hide-labels', default=False, action='store_true', help='sembunyikan label')  # argumen sembunyikan label
    parser.add_argument('--hide-conf', default=False, action='store_true', help='sembunyikan kepercayaan')  # argumen sembunyikan kepercayaan
    parser.add_argument('--half', action='store_true', help='gunakan inferensi FP16 half-precision')  # argumen gunakan inferensi FP16
    opt = parser.parse_args()  # mengurai argumen
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # memperluas ukuran gambar jika hanya satu dimensi
    return opt  # mengembalikan argumen yang diurai

def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))  # mencetak argumen yang diurai
    check_requirements(exclude=('tensorboard', 'thop'))  # memeriksa persyaratan
    run(**vars(opt))  # menjalankan fungsi run dengan argumen yang diurai

if __name__ == "__main__":
    opt = parse_opt()  # mengurai argumen
    main(opt)  # menjalankan fungsi utama dengan argumen yang diurai


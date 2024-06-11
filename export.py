# YOLOv5 🚀 oleh Ultralytics, lisensi GPL-3.0
"""
Ekspor model PyTorch ke format TorchScript, ONNX, CoreML

Penggunaan:
    $ python path/to/export.py --weights yolov5s.pt --img 640 --batch 1
"""

import argparse  # Mengimpor modul argparse untuk menangani argumen baris perintah
import sys  # Mengimpor modul sys untuk memanipulasi sistem Python
import time  # Mengimpor modul time untuk mengukur waktu
from pathlib import Path  # Mengimpor Path dari modul pathlib untuk manipulasi path

import torch  # Mengimpor pustaka PyTorch untuk machine learning
import torch.nn as nn  # Mengimpor modul neural network dari PyTorch
from torch.utils.mobile_optimizer import optimize_for_mobile  # Mengimpor optimisasi untuk perangkat mobile dari PyTorch

FILE = Path(__file__).absolute()  # Mendapatkan path absolut dari file skrip ini
sys.path.append(FILE.parents[0].as_posix())  # Menambahkan direktori induk dari file ini ke path

from models.common import Conv  # Mengimpor kelas Conv dari models.common
from models.yolo import Detect  # Mengimpor kelas Detect dari models.yolo
from models.experimental import attempt_load  # Mengimpor fungsi attempt_load dari models.experimental
from utils.activations import Hardswish, SiLU  # Mengimpor aktivasi Hardswish dan SiLU dari utils.activations
from utils.general import colorstr, check_img_size, check_requirements, file_size, set_logging  # Mengimpor fungsi utilitas dari utils.general
from utils.torch_utils import select_device  # Mengimpor fungsi select_device dari utils.torch_utils

def export_torchscript(model, img, file, optimize):
    # Ekspor model ke TorchScript
    prefix = colorstr('TorchScript:')  # Prefix untuk pesan konsol
    try:
        print(f'\n{prefix} mulai ekspor dengan torch {torch.__version__}...')
        f = file.with_suffix('.torchscript.pt')  # Menambahkan suffix .torchscript.pt ke file
        ts = torch.jit.trace(model, img, strict=False)  # Melacak model dengan TorchScript
        (optimize_for_mobile(ts) if optimize else ts).save(f)  # Menyimpan model yang dioptimalkan atau model asli
        print(f'{prefix} ekspor sukses, disimpan sebagai {f} ({file_size(f):.1f} MB)')  # Pesan sukses
        return ts  # Mengembalikan model TorchScript
    except Exception as e:  # Menangani pengecualian
        print(f'{prefix} ekspor gagal: {e}')  # Pesan kegagalan

def export_onnx(model, img, file, opset, train, dynamic, simplify):
    # Ekspor model ke ONNX
    prefix = colorstr('ONNX:')  # Prefix untuk pesan konsol
    try:
        check_requirements(('onnx', 'onnx-simplifier'))  # Memeriksa apakah onnx dan onnx-simplifier terinstal
        import onnx  # Mengimpor pustaka ONNX

        print(f'\n{prefix} mulai ekspor dengan onnx {onnx.__version__}...')
        f = file.with_suffix('.onnx')  # Menambahkan suffix .onnx ke file
        torch.onnx.export(model, img, f, verbose=False, opset_version=opset,
                          training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=not train,
                          input_names=['images'],
                          output_names=['output'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                        'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                                        } if dynamic else None)

        # Memeriksa model ONNX
        model_onnx = onnx.load(f)  # Memuat model ONNX
        onnx.checker.check_model(model_onnx)  # Memeriksa model ONNX
        # print(onnx.helper.printable_graph(model_onnx.graph))  # Mencetak grafis model ONNX

        # Menyederhanakan model ONNX
        if simplify:
            try:
                import onnxsim  # Mengimpor pustaka onnx-simplifier

                print(f'{prefix} menyederhanakan dengan onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'images': list(img.shape)} if dynamic else None)
                assert check, 'assert check failed'  # Memastikan penyederhanaan berhasil
                onnx.save(model_onnx, f)  # Menyimpan model ONNX yang disederhanakan
            except Exception as e:  # Menangani pengecualian
                print(f'{prefix} penyederhanaan gagal: {e}')
        print(f'{prefix} ekspor sukses, disimpan sebagai {f} ({file_size(f):.1f} MB)')  # Pesan sukses
        print(f"{prefix} jalankan --dynamic ONNX model inference dengan: 'python detect.py --weights {f}'")  # Petunjuk menjalankan model
    except Exception as e:  # Menangani pengecualian
        print(f'{prefix} ekspor gagal: {e}')  # Pesan kegagalan

def export_coreml(model, img, file):
    # Ekspor model ke CoreML
    prefix = colorstr('CoreML:')  # Prefix untuk pesan konsol
    try:
        check_requirements(('coremltools',))  # Memeriksa apakah coremltools terinstal
        import coremltools as ct  # Mengimpor pustaka coremltools

        print(f'\n{prefix} mulai ekspor dengan coremltools {ct.__version__}...')
        f = file.with_suffix('.mlmodel')  # Menambahkan suffix .mlmodel ke file
        model.train()  # Model CoreML harus berada dalam mode train
        ts = torch.jit.trace(model, img, strict=False)  # Melacak model dengan TorchScript
        model = ct.convert(ts, inputs=[ct.ImageType('image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])  # Mengonversi model ke CoreML
        model.save(f)  # Menyimpan model CoreML
        print(f'{prefix} ekspor sukses, disimpan sebagai {f} ({file_size(f):.1f} MB)')  # Pesan sukses
    except Exception as e:  # Menangani pengecualian
        print(f'\n{prefix} ekspor gagal: {e}')  # Pesan kegagalan

def run(weights='./yolov5s.pt',  # path ke weights
        img_size=(640, 640),  # ukuran gambar (tinggi, lebar)
        batch_size=1,  # ukuran batch
        device='cpu',  # perangkat cuda, misalnya 0 atau 0,1,2,3 atau cpu
        include=('torchscript', 'onnx', 'coreml'),  # format yang akan disertakan
        half=False,  # ekspor FP16 half-precision
        inplace=False,  # set YOLOv5 Detect() inplace=True
        train=False,  # mode model.train()
        optimize=False,  # TorchScript: optimisasi untuk mobile
        dynamic=False,  # ONNX: sumbu dinamis
        simplify=False,  # ONNX: menyederhanakan model
        opset=12,  # ONNX: versi opset
        ):
    t = time.time()  # Mengambil waktu saat ini
    include = [x.lower() for x in include]  # Mengubah format yang disertakan menjadi huruf kecil
    img_size *= 2 if len(img_size) == 1 else 1  # Mengembangkan ukuran gambar jika hanya satu dimensi
    file = Path(weights)  # Membuat objek Path untuk weights

    # Memuat model PyTorch
    device = select_device(device)  # Memilih perangkat (CPU atau GPU)
    assert not (device.type == 'cpu' and half), '--half hanya kompatibel dengan ekspor GPU, gunakan --device 0'  # Memastikan half hanya digunakan dengan GPU
    model = attempt_load(weights, map_location=device)  # Memuat model FP32
    names = model.names  # Mengambil nama-nama kelas dari model

    # Input
    gs = int(max(model.stride))  # Ukuran grid (stride maksimal)
    img_size = [check_img_size(x, gs) for x in img_size]  # Memastikan ukuran gambar merupakan kelipatan gs
    img = torch.zeros(batch_size, 3, *img_size).to(device)  # Membuat tensor gambar dengan ukuran tertentu

    # Memperbarui model
    if half:
        img, model = img.half(), model.half()  # Mengubah gambar dan model ke FP16
    model.train() if train else model.eval()  # Mode training = tidak ada konstruksi grid layer Detect()
    for k, m in model.named_modules():
        if isinstance(m, Conv):  # Menetapkan aktivasi yang ramah ekspor
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic
            # m.forward = m.forward_export  # Menetapkan forward (opsional)

    for _ in range(2):
        y = model(img)  # Dry runs
    print(f"\n{colorstr('PyTorch:')} memulai dari {weights} ({file_size(weights):.1f} MB)")  # Pesan memulai

    # Ekspor
    if 'torchscript' in include:
        export_torchscript(model, img, file, optimize)
    if 'onnx' in include:
        export_onnx(model, img, file, opset, train, dynamic, simplify)
    if 'coreml' in include:
        export_coreml(model, img, file)

    # Selesai
    print(f'\nEkspor selesai ({time.time() - t:.2f}s)'
          f"\nHasil disimpan ke {colorstr('bold', file.parent.resolve())}"
          f'\nVisualisasikan dengan https://netron.app')

def parse_opt():
    parser = argparse.ArgumentParser()  # Membuat parser untuk argumen baris perintah
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='path ke weights')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='ukuran gambar (tinggi, lebar)')
    parser.add_argument('--batch-size', type=int, default=1, help='ukuran batch')
    parser.add_argument('--device', default='cpu', help='perangkat cuda, misalnya 0 atau 0,1,2,3 atau cpu')
    parser.add_argument('--include', nargs='+', default=['torchscript', 'onnx', 'coreml'], help='format yang disertakan')
    parser.add_argument('--half', action='store_true', help='ekspor FP16 half-precision')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--train', action='store_true', help='mode model.train()')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimisasi untuk mobile')
    parser.add_argument('--dynamic', action='store_true', help='ONNX: sumbu dinamis')
    parser.add_argument('--simplify', action='store_true', help='ONNX: menyederhanakan model')
    parser.add_argument('--opset', type=int, default=13, help='ONNX: versi opset')
    opt = parser.parse_args()  # Mengurai argumen baris perintah
    return opt  # Mengembalikan argumen yang diurai

def main(opt):
    set_logging()  # Mengatur logging
    print(colorstr('export: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))  # Mencetak argumen yang diurai
    run(**vars(opt))  # Menjalankan fungsi run dengan argumen yang diurai

if __name__ == "__main__":
    opt = parse_opt()  # Mengurai argumen baris perintah
    main(opt)  # Menjalankan fungsi utama dengan argumen yang diurai

import os
import csv
import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import linalg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# ==================== Configuration ====================
REAL_FOLDER = r"C:\Users\11321\Desktop\LLMs\HV\data\2019\img"
FAKE_FOLDER = r"C:\Users\11321\Desktop\LLMs\HG\Main\model\GPT\2019"
RESULT_CSV = "./gpt_2019_fid_results.csv"
LOG_FILE = "./gpt_2019_fid_calculate.log"

BATCH_SIZE = 32
DEVICE = "cuda"
FEATURE_DIMS = 2048
NUM_WORKERS = 0
# Official pytorch-fid weight URL
FID_WEIGHTS_URL = (
    "https://github.com/mseitzer/pytorch-fid/releases/download/"
    "fid_weights/pt_inception-2015-12-05-6726825d.pth"
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def setup_logger(log_path: str):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


class InceptionV3(nn.Module):
    DEFAULT_BLOCK_INDEX = 3

    BLOCK_INDEX_BY_DIM = {
        64: 0,
        192: 1,
        768: 2,
        2048: 3,
    }

    def __init__(
        self,
        output_blocks: Sequence[int] = (DEFAULT_BLOCK_INDEX,),
        resize_input: bool = True,
        normalize_input: bool = True,
        requires_grad: bool = False,
        use_fid_inception: bool = True,
    ):
        super().__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        if self.last_needed_block > 3:
            raise ValueError("Last output block index cannot exceed 3.")

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = _inception_v3(pretrained=True)

        # Block 0
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ]
            self.blocks.append(nn.Sequential(*block3))

        for parameter in self.parameters():
            parameter.requires_grad = requires_grad

    def forward(self, inp: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        x = inp

        if self.resize_input:
            x = F.interpolate(
                x,
                size=(299, 299),
                mode="bilinear",
                align_corners=False,
            )

        if self.normalize_input:
            x = 2.0 * x - 1.0

        for index, block in enumerate(self.blocks):
            x = block(x)

            if index in self.output_blocks:
                outputs.append(x)

            if index == self.last_needed_block:
                break

        return outputs


def _inception_v3(*args, **kwargs):
    try:
        version = tuple(
            int(part) for part in torchvision.__version__.split("+")[0].split(".")[:2]
        )
    except (TypeError, ValueError):
        version = (0,)

    if version >= (0, 6):
        kwargs.setdefault("init_weights", False)

    if version >= (0, 13) and "pretrained" in kwargs:
        pretrained = kwargs.pop("pretrained")
        if pretrained:
            from torchvision.models import Inception_V3_Weights
            kwargs["weights"] = Inception_V3_Weights.DEFAULT
        else:
            kwargs["weights"] = None

    return torchvision.models.inception_v3(*args, **kwargs)


def fid_inception_v3() -> nn.Module:
    inception = _inception_v3(
        num_classes=1008,
        aux_logits=False,
        pretrained=False,
    )

    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)

    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)

    inception.Mixed_7b = FIDInceptionE1(1280)
    inception.Mixed_7c = FIDInceptionE2(2048)

    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)

    return inception


class FIDInceptionA(torchvision.models.inception.InceptionA):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            dim=1,
        )


class FIDInceptionC(torchvision.models.inception.InceptionC):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            dim=1,
        )


class FIDInceptionE1(torchvision.models.inception.InceptionE):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = torch.cat(
            [
                self.branch3x3_2a(branch3x3),
                self.branch3x3_2b(branch3x3),
            ],
            dim=1,
        )

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = torch.cat(
            [
                self.branch3x3dbl_3a(branch3x3dbl),
                self.branch3x3dbl_3b(branch3x3dbl),
            ],
            dim=1,
        )

        branch_pool = F.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            dim=1,
        )


class FIDInceptionE2(torchvision.models.inception.InceptionE):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = torch.cat(
            [
                self.branch3x3_2a(branch3x3),
                self.branch3x3_2b(branch3x3),
            ],
            dim=1,
        )

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = torch.cat(
            [
                self.branch3x3dbl_3a(branch3x3dbl),
                self.branch3x3dbl_3b(branch3x3dbl),
            ],
            dim=1,
        )

        branch_pool = F.max_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            dim=1,
        )


class ImagePathDataset(Dataset):

    def __init__(self, files: Sequence[Path]):
        if not files:
            raise ValueError("Image file list is empty.")

        self.files = list(files)

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.files[index]

        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                return self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {path}") from exc


def collect_image_files(directory: str) -> List[Path]:

    root = Path(directory).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not files:
        raise ValueError(f"No supported images found in directory: {root}")

    return files


@torch.no_grad()
def get_activations(
    files: Sequence[Path],
    model: nn.Module,
    batch_size: int,
    dims: int,
    device: torch.device,
    num_workers: int = 0,
) -> np.ndarray:

    if len(files) < 2:
        raise ValueError("At least 2 images are required to compute covariance.")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    batch_size = min(batch_size, len(files))

    dataset = ImagePathDataset(files)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    activations = np.empty((len(files), dims), dtype=np.float64)
    model.eval()

    offset = 0
    total_batches = len(dataloader)

    for batch_index, batch in enumerate(dataloader, start=1):
        batch = batch.to(device=device, dtype=torch.float32, non_blocking=True)

        prediction = model(batch)[0]

        if prediction.shape[2] != 1 or prediction.shape[3] != 1:
            prediction = F.adaptive_avg_pool2d(prediction, output_size=(1, 1))

        prediction = prediction.squeeze(3).squeeze(2)
        prediction_np = prediction.cpu().numpy().astype(np.float64, copy=False)

        next_offset = offset + prediction_np.shape[0]
        activations[offset:next_offset] = prediction_np
        offset = next_offset

        if batch_index % 10 == 0 or batch_index == total_batches:
            logging.info(f"  Feature extraction progress: {batch_index}/{total_batches} batches")

    return activations


def calculate_activation_statistics(
    files: Sequence[Path],
    model: nn.Module,
    batch_size: int,
    dims: int,
    device: torch.device,
    num_workers: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:

    activations = get_activations(
        files=files,
        model=model,
        batch_size=batch_size,
        dims=dims,
        device=device,
        num_workers=num_workers,
    )

    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    return mu, sigma


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError(f"Mean vectors have mismatched shapes: {mu1.shape} vs {mu2.shape}")

    if sigma1.shape != sigma2.shape:
        raise ValueError(f"Covariance matrices have mismatched shapes: {sigma1.shape} vs {sigma2.shape}")

    diff = mu1 - mu2

    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if not np.isfinite(covmean).all():
        logging.warning("Covariance product is nearly singular, adding epsilon for sqrtm")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        max_imaginary = np.max(np.abs(np.imag(covmean)))

        if max_imaginary > 1e-3:
            raise ValueError(f"Matrix sqrt produced large imaginary part: {max_imaginary}")

        covmean = np.real(covmean)

    fid = (
        diff.dot(diff)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2.0 * np.trace(covmean)
    )

    if fid < 0 and abs(fid) < 1e-6:
        fid = 0.0

    return float(fid)


def main():
    logger = setup_logger(LOG_FILE)
    logger.info("=" * 60)
    logger.info("FID Calculation Program Started")
    logger.info(f"Real image folder: {os.path.abspath(REAL_FOLDER)}")
    logger.info(f"Fake image folder: {os.path.abspath(FAKE_FOLDER)}")
    logger.info(f"Result CSV path: {os.path.abspath(RESULT_CSV)}")
    logger.info(f"Log file path: {os.path.abspath(LOG_FILE)}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Feature dimensions: {FEATURE_DIMS}")
    logger.info("=" * 60)

    # Validate paths
    if not os.path.isdir(REAL_FOLDER):
        logger.error(f"Real image folder does not exist: {REAL_FOLDER}")
        return
    if not os.path.isdir(FAKE_FOLDER):
        logger.error(f"Fake image folder does not exist: {FAKE_FOLDER}")
        return

    # Auto select device
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is not available, falling back to CPU")
        target_device = torch.device("cpu")
    else:
        target_device = torch.device(DEVICE)
    logger.info(f"Running on device: {target_device}")

    # Collect image files
    try:
        real_files = collect_image_files(REAL_FOLDER)
        fake_files = collect_image_files(FAKE_FOLDER)
    except Exception as e:
        logger.error(f"Failed to collect image files: {str(e)}")
        return

    logger.info(f"Found {len(real_files)} real images")
    logger.info(f"Found {len(fake_files)} fake images")

    # Validate feature dimension
    if FEATURE_DIMS not in InceptionV3.BLOCK_INDEX_BY_DIM:
        logger.error(f"Unsupported feature dimension: {FEATURE_DIMS}")
        return

    # Load model
    logger.info("Loading InceptionV3 model...")
    block_index = InceptionV3.BLOCK_INDEX_BY_DIM[FEATURE_DIMS]
    model = InceptionV3(
        output_blocks=(block_index,),
        resize_input=True,
        normalize_input=True,
        requires_grad=False,
        use_fid_inception=True,
    ).to(target_device)
    logger.info("Model loaded successfully")

    # Compute statistics
    logger.info("Computing statistics for real images...")
    try:
        mu_real, sigma_real = calculate_activation_statistics(
            files=real_files,
            model=model,
            batch_size=BATCH_SIZE,
            dims=FEATURE_DIMS,
            device=target_device,
            num_workers=NUM_WORKERS,
        )
    except Exception as e:
        logger.error(f"Failed to compute real image statistics: {str(e)}")
        return
    logger.info("Real image statistics computed")

    logger.info("Computing statistics for fake images...")
    try:
        mu_fake, sigma_fake = calculate_activation_statistics(
            files=fake_files,
            model=model,
            batch_size=BATCH_SIZE,
            dims=FEATURE_DIMS,
            device=target_device,
            num_workers=NUM_WORKERS,
        )
    except Exception as e:
        logger.error(f"Failed to compute fake image statistics: {str(e)}")
        return
    logger.info("Fake image statistics computed")

    # Calculate FID
    logger.info("Calculating Fréchet Inception Distance...")
    try:
        fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    except Exception as e:
        logger.error(f"Failed to calculate FID: {str(e)}")
        return

    # Save results to CSV (Chinese headers and remarks)
    logger.info("Saving results to CSV...")
    try:
        with open(RESULT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['指标项', '数值/路径', '备注'])
            writer.writerow(['真实图像文件夹', os.path.abspath(REAL_FOLDER), ''])
            writer.writerow(['生成图像文件夹', os.path.abspath(FAKE_FOLDER), ''])
            writer.writerow(['真实图像数量', len(real_files), ''])
            writer.writerow(['生成图像数量', len(fake_files), ''])
            writer.writerow(['计算设备', str(target_device), ''])
            writer.writerow(['特征维度', FEATURE_DIMS, ''])
            writer.writerow(['FID 值', round(fid_value, 6), '越小越好，0表示分布完全一致'])
        logger.info(f"Result CSV saved to: {os.path.abspath(RESULT_CSV)}")
    except Exception as e:
        logger.error(f"Failed to save CSV file: {str(e)}")

    # Final summary
    logger.info("=" * 60)
    logger.info("FID calculation completed")
    logger.info(f"Final FID score: {fid_value:.6f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

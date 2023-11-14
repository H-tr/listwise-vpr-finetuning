# Visual Place Recognition (VPR) Fine-tuning without GPS Labels

## Project Overview

This research project aims to enhance Visual Place Recognition (VPR) in indoor and semi-indoor environments where GPS signals are weak and noisy. The proposed method utilizes a listwise loss, such as Average Precision (AP), to fine-tune VPR models without relying on GPS labels. The motivation behind this approach is to address the challenges posed by inaccurate or absent GPS information in certain environments, where accuracy requirements are higher than traditional outdoor VPR scenarios.

## Features

- **No Dependency on GPS Labels:** The algorithm fine-tunes VPR models without the need for accurate GPS labels, making it suitable for indoor or semi-indoor environments with weak and noisy GPS signals.

- **Automatic Fine-tuning:** Users only need to input a video without any additional preprocessing. The algorithm automatically fine-tunes the VPR model, adapting to the specific characteristics of the provided data.

- **Improved Accuracy in Complex Areas:** In areas with complex structures, such as multiple layers or varying heights, traditional GPS labels might be misleading. The proposed method considers these complexities to differentiate between locations that might appear similar but represent distinct places.

## How to Use

### Installation

1. Clone the repository:

   ```bash
   https://github.com/H-tr/listwise-vpr-finetuning.git
   cd listwise-vpr-finetuning
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Place your input video in the designated folder (provide path if applicable).

2. Run the fine-tuning script:

   ```bash
   python application.py --input_video_path /path/to/your/video
   ```

## Results

[Include any key findings, performance metrics, or visualizations that showcase the effectiveness of the proposed method.]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [NetVLAD PyTorch](https://github.com/Nanne/pytorch-NetVlad/blob/master/netvlad.py)

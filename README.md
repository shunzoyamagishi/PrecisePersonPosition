# MyProject

This repository includes code that is **adapted from**  
[Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2).

## install
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
mv Depth-Anything-V2 Depth_Anything_V2 && cd Depth_Anything_V2
pip install -r requirement.txt
```
- Download checkpoints from DepthAnythingV2 repogitory and put in "checkpoints" directory.
- Refer to the [Depth-Anything-V2 repository](https://github.com/DepthAnything/Depth-Anything-V2), download the checkpoint file from that repository, and place it in checkpoint directory.


## Train
```bash
python train.py --config config/default.yaml
```
学習結果は`trains/yyyy-mm-dd_hh-mm-ss`に  
- `config.yaml` (実行したconfigファイルのコピー)
- `train.log`  (ログファイル)
- `events.out.tfevent`ファイル
- 学習途中のweight  

が保存されます。  

学習のログを確認するには
```bash
tensorboard --logdir=trains/yyyy-mm-dd_hh-mm-ss
```

## Reference
```bash
python reference.py --path image/video path --config config/default_reference.yaml
```



## License

This project follows different licensing conditions depending on the model used:

- **Depth-Anything-V2-Small**  
  Licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

- **Depth-Anything-V2-Base / Large / Giant**  
  Licensed under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/)  
  (Non-commercial use only)

---

### This Project

If you use this repository with the **Small model**, the project may be used under Apache-2.0.  
If you use this repository with the **Base / Large / Giant models**, the project must follow CC-BY-NC-4.0.

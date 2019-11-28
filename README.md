# step1

```bash
# terminal 1
bash deploy 192.168.1.2 local
ssh HwHiAiUser@192.168.1.2
cd ~/HIAI_PROJECTS/ascend_workspace/segmentation/out
./ascend_segmentation
```

# step2

```bash
# terminal 2
bash getResult.sh
```
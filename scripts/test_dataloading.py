from world_model.opendv.tokenized_sequence_opendv import TokenizedSequenceOpenDVDataModule


dm = TokenizedSequenceOpenDVDataModule(
    data_root_dir="$ycy_ALL_CCFRSCRATCH/OpenDV_release/tokens",
    video_list_path="$ycy_ALL_CCFRSCRATCH/OpenDV_release/video_list.txt",
    val_video_list_path="$ycy_ALL_CCFRSCRATCH/OpenDV_release/val_video_list.txt",
    batch_size=16,
    num_workers=8,
).setup()

train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()

for idx, batch in enumerate(train_loader):
    print(batch)
    if idx == 2:
        break

for idx, batch in enumerate(val_loader):
    print(batch)
    if idx == 2:
        break

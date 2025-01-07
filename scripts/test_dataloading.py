from world_model.opendv.tokenized_sequence_opendv import TokenizedSequenceOpenDVDataModule


dm = TokenizedSequenceOpenDVDataModule(
    data_root_dir="$fzh_ALL_CCFRSCRATCH/OpenDV_processed/flat_tokens",
    video_list_path="$fzh_ALL_CCFRSCRATCH/OpenDV_processed/train.json",
    val_video_list_path="$fzh_ALL_CCFRSCRATCH/OpenDV_processed/val.json",
    sequence_length=20,
    batch_size=16,
    num_workers=8,
).setup()

train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()

for idx, batch in enumerate(train_loader):
    print(batch["visual_tokens"].shape)
    if idx == 10:
        break

for idx, batch in enumerate(val_loader):
    print(batch["visual_tokens"].shape)
    if idx == 10:
        break

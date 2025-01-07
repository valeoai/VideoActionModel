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
    print(batch["visual_tokens"].shape, batch["idx"][:3])
    if idx == 5:
        state_dict = dm.state_dict()
    if idx == 10:
        break

for idx, _ in enumerate(val_loader):
    if idx == 10:
        break


dm2 = TokenizedSequenceOpenDVDataModule(
    data_root_dir="$fzh_ALL_CCFRSCRATCH/OpenDV_processed/flat_tokens",
    video_list_path="$fzh_ALL_CCFRSCRATCH/OpenDV_processed/train.json",
    val_video_list_path="$fzh_ALL_CCFRSCRATCH/OpenDV_processed/val.json",
    sequence_length=20,
    batch_size=16,
    num_workers=8,
).setup()
dm2.load_state_dict(state_dict)

train_loader_2 = dm2.train_dataloader()

for idx, batch in enumerate(train_loader_2):
    print(batch["visual_tokens"].shape, batch["idx"][:3])
    if idx == 5:
        break

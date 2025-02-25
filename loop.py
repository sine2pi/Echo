
def load_wave(wave_path, sample_rate: int = 16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    return waveform

class audioDataset(Dataset):
    def __init__(self, csv_file, aud_dir, tokenizer, sample_rate=16000):
        self.aud_dir = aud_dir
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.samples = []

        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) 
            for row in reader:
                aud_path, label = row[0], row[1]
                self.samples.append((aud_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        aud_path, label = self.samples[idx]
        label = handle_unknown_characters(label)
        aud = f'{self.aud_dir}/{aud_path}'
        return {
            'input_features': aud,
            'labels': label,
            'input_ids': label 
        }

def handle_unknown_characters(label): 
    label = label.encode('utf-8').decode('utf-8', errors='replace')
    label = neologdn.normalize(label, repeat=1)
    return label

class DataCollatorWithPadding:
    def __init__(self, tokenizer, n_mels, n_fft, hop_length, sample_rate=16000):
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mel_spectrogram_transform = transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            hop_length=self.hop_length
        )

    def __call__(self, features):
        input_features, dec_input_ids, labels = [], [], []

        for f in features:
            aud_path = f['input_features']
            aud, _ = torchaudio.load(aud_path, normalize=True)
            aud = whisper.pad_or_trim(aud.flatten())

            mel_spectrogram = self.mel_spectrogram_transform(aud)
            log_mel_spectrogram = torch.log(mel_spectrogram + 1e-8)  

            label = handle_unknown_characters(f['labels']) 
            encoded_input = self.tokenizer.encode(label)
            encoded_label = self.tokenizer.encode(label)

            dec_input_ids.append([self.tokenizer.bos_token_id] + encoded_input)
            labels.append(encoded_label + [self.tokenizer.eos_token_id])
            input_features.append(log_mel_spectrogram)

        input_features = torch.stack(input_features)

        input_lengths = [len(ids) for ids in dec_input_ids]
        label_lengths = [len(lab) for lab in labels]
        max_len = max(input_lengths + label_lengths)

        dec_input_ids = [np.pad(ids, (0, max_len - len(ids)), 'constant', constant_values=self.tokenizer.pad_token_id) for ids in dec_input_ids]
        labels = [np.pad(lab, (0, max_len - len(lab)), 'constant', constant_values=-100) for lab in labels]

        batch = {
            "input_ids": dec_input_ids,
            "labels": labels,
            "input_features": input_features
        }
        batch = {k: torch.tensor(v, requires_grad=False) for k, v in batch.items()}
        return batch

metrics_cer = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred["predictions"]
    label_ids = pred["label_ids"]
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    cer = 100 * metrics_cer.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

checkpoint_dir = 'D:/newproject/test/'
os.makedirs(checkpoint_dir, exist_ok=True)
log_dir = "D:/newproject/test/logs_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)

writer = SummaryWriter(log_dir)
import logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'training.log'), 
    filemode='w', 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)

def train_and_evaluate(model, train_loader, eval_loader, optimizer, scheduler, loss_fn, num_epochs=1, max_steps=None, device='cuda', accumulation_steps=1, clear_cache=True, log_interval=10, eval_interval=20, save_interval=100, checkpoint_dir="checkpoint_dir", log_dir="log_dir"):
    model.to(device)
    global_step = 0
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=log_dir)
    lr_warning_printed = False

    for epoch in range(num_epochs):
        if max_steps is not None and global_step >= max_steps:
            break

        model.train()
        total_loss = 0
        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(progress_bar):
            if max_steps is not None and global_step >= max_steps:
                break

            start_time = time.time()

            input_features = batch['input_features'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].long().to(device)

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_training"):
                    with torch.amp.autocast(device_type='cuda'):
                        input_features_encoded = model.encoder(input_features)
                        decoder_output = model.decoder(input_ids, input_features_encoded)
                        logits = decoder_output.view(-1, decoder_output.size(-1))
                        loss = loss_fn(logits, labels.view(-1))
                        total_loss += loss.item()
                        loss = loss / accumulation_steps

                    scaler.scale(loss).backward()

                    if (step + 1) % accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                        if clear_cache:
                            torch.cuda.empty_cache()

            global_step += 1
            end_time = time.time()
            samples_per_sec = len(batch['input_features']) / (end_time - start_time)

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)

            if global_step % log_interval == 0:
                writer.add_scalar('Loss/train', total_loss / (step + 1), global_step)
                writer.add_scalar('GradientNorm', total_norm, global_step)
                
                lr = optimizer.param_groups[0].get('lr', None)
                if lr is not None:
                    writer.add_scalar('LearningRate', lr, global_step)
                else:
                    if not lr_warning_printed:
                        print(f"Warning: Learning rate is None at step {global_step}")
                        lr_warning_printed = True

                writer.add_scalar('SamplesPerSec', samples_per_sec, global_step)

            if global_step % eval_interval == 0:
                model.eval()
                eval_loss = 0
                all_predictions = []
                all_labels = []
                with torch.no_grad():
                    for eval_batch in eval_loader:
                        input_features = eval_batch['input_features'].to(device)
                        input_ids = eval_batch['input_ids'].to(device)
                        labels = eval_batch['labels'].long().to(device)
                        input_features_encoded = model.encoder(input_features)
                        decoder_output = model.decoder(input_ids, input_features_encoded)
                        logits = decoder_output.view(-1, decoder_output.size(-1))
                        loss = loss_fn(logits, labels.view(-1))
                        eval_loss += loss.item()
                        all_predictions.extend(torch.argmax(decoder_output, dim=-1).cpu().numpy().tolist())
                        all_labels.extend(labels.cpu().numpy().tolist())

                eval_loss /= len(eval_loader)
                predictions = {"predictions": np.array(all_predictions, dtype="object"), "label_ids": np.array(all_labels, dtype="object")}
                metrics = compute_metrics(predictions)
                writer.add_scalar('Loss/eval', eval_loss, global_step)
                writer.add_scalar('CER', metrics['cer'], global_step)
                scheduler.step()  # Step the scheduler

                sample_indices = range(min(1, len(all_predictions))) 
                for idx in sample_indices:
                    pred_str = tokenizer.decode(all_predictions[idx], skip_special_tokens=True)
                    label_str = tokenizer.decode(all_labels[idx], skip_special_tokens=True)
                    print(f"Evaluation Loss: {eval_loss:.4f}")
                    print(f"Evaluation Sample {idx}: Prediction: {pred_str}, Label: {label_str}")
                    logging.info(f"Evaluation Sample {idx}: Prediction: {pred_str}, Label: {label_str}")

                model.train()

                print(f"Evaluation Loss: {eval_loss:.4f}")
                print(f"Character Error Rate (CER): {metrics['cer']:.4f}")

            if global_step % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{global_step}.pt')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model saved at step {global_step} to {checkpoint_path}")
                logging.info(f"Model saved at step {global_step} to {checkpoint_path}")

        scheduler.step()

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')
        logging.info(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

    final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    logging.info(f"Final model saved to {final_model_path}")
    writer.close()


if __name__ == "__main__":

    tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-small")
    csv_file = 'D:/proj/datasets/gf_1/metadata.csv'
    audio_dir = 'D:/proj/datasets/gf_1/'

    def train_val_dataset(dataset, val_split=0.001):
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
        datasets = {}
        datasets['train'] = Subset(dataset, train_idx)
        datasets['val'] = Subset(dataset, val_idx)
        return datasets

    dataset = audioDataset(csv_file, audio_dir, tokenizer)
    datasets = train_val_dataset(dataset)
    train_dataset = datasets['train']
    eval_dataset = datasets['val']
    
    def train_dataloader():   
        return DataLoader(
            train_dataset,
            batch_size=1,
            drop_last=False, 
            shuffle=True, 
            num_workers=0,
            collate_fn=collate_fn
        )

    def eval_dataloader():
        return DataLoader(
            eval_dataset,
            batch_size=1, 
            drop_last=True,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

    collate_fn = DataCollatorWithPadding(tokenizer, n_fft=1024, hop_length=256, n_mels=80)
    train_loader = train_dataloader()
    eval_loader = eval_dataloader()

    config = Config(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=1024,
        n_audio_head=16,
        n_audio_layer=24,
        n_vocab=51865,
        n_text_ctx=448,
        n_text_state=1024,
        n_text_head=16,
        n_text_layer=20,
        checkpointing=True,
        )

    model = model(config).cuda()
    # model.resize_token_embeddings(len(tokenizer))
    optimizer = transformers.Adafactor(model.parameters(), 
                                    clip_threshold=0.99, 
                                    weight_decay=0.025, 
                                    scale_parameter=True, 
                                    relative_step=False, 
                                    warmup_init=False, 
                                    lr=2.25e-3)

    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    from torch.utils.tensorboard import SummaryWriter

    train_and_evaluate(model, train_loader, eval_loader, optimizer, scheduler, loss_fn, max_steps=100, num_epochs=1, device='cuda', accumulation_steps=1, clear_cache=True, log_interval=1, eval_interval=10, save_interval=100, checkpoint_dir=checkpoint_dir, log_dir=log_dir)

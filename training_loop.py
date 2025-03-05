extractor = WhisperFeatureExtractor.from_pretrained(
    pretrained_model_name_or_path="openai/whisper-small", 
    feature_size=128, sample_rate=16000, do_normalize=True)

tokenizer = WhisperTokenizerFast.from_pretrained(
    pretrained_model_name_or_path="openai/whisper-small", 
    language="en", task="transcribe")

processor = WhisperProcessor.from_pretrained(
    pretrained_model_name_or_path="openai/whisper-small", 
    feature_extractor=extractor,
    tokenizer=tokenizer)
    
def process_fn(batch):
    return prepare_dataset(batch=batch, extractor=extractor, tokenizer=tokenizer)
    
def prepare_dataset(batch, extractor, tokenizer):
    audio = batch["audio"]
    batch["input_features"] = extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

def prepare_dataset_with_columns(batch, extractor, tokenizer):
    result = {}
    audio = batch["audio"]
    result["input_features"] = extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    result["labels"] = tokenizer(batch["sentence"]).input_ids
    return result
    
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    extractor: Any
    tokenizer: Any
    decoder_start_token_id=50258
    pad_token_id=50257

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
      
        batch["input_ids"] = Echo.shift_tokens_right(
            input_ids=labels,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id
        )
        return batch

metric = evaluate.load(path="wer")

def compute_metrics(pred, tokenizer):
    pred_ids = pred["predictions"]
    label_ids = pred["label_ids"]
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    else:
        pred_ids = pred_ids
    if pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str) # type: ignore
    return {"wer": wer}
def train_and_evaluate(model, tokenizer, train_loader, eval_loader, optimizer, scheduler, loss_fn, 
                      max_steps=10000, device='cuda', accumulation_steps=1, clear_cache=True, 
                      log_interval=10, eval_interval=100, save_interval=1000, 
                      checkpoint_dir="checkpoint_dir", log_dir="log_dir"):
    model.to(device)
    global_step = 0
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=log_dir)
    train_iterator = iter(train_loader)
    total_loss = 0
    step_in_report = 0
    dataset_epochs = 0
    
    progress_bar = tqdm(total=max_steps, desc="Training")
    
    model.train()
    optimizer.zero_grad()
    
    while global_step < max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
            
            dataset_epochs += 1
            print(f"Starting dataset epoch {dataset_epochs}")
            
            if step_in_report > 0:
                avg_loss = total_loss / step_in_report if step_in_report > 0 else 0
                logging.info(f"Dataset iteration complete - Steps: {global_step}, Avg Loss: {avg_loss:.4f}")
                total_loss = 0
                step_in_report = 0
        
        start_time = time.time()

        input_features = batch['input_features'].to(device)
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].long().to(device)
        
        with torch.autocast(device_type='cuda'):
            if global_step % 100 == 0:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                    with record_function("model_training"):
                        input_features_encoded = model.encoder(input_features)
                        decoder_output = model.decoder(input_ids, input_features_encoded)
            else:
                input_features_encoded = model.encoder(input_features)
                decoder_output = model.decoder(input_ids, input_features_encoded)
        
        logits = decoder_output.view(-1, decoder_output.size(-1))
        
        active_logits = logits.view(-1, decoder_output.size(-1))
        active_labels = labels.view(-1)
        active_mask = active_labels != tokenizer.pad_token_id
        active_logits = active_logits[active_mask]
        active_labels = active_labels[active_mask]
        loss = loss_fn(active_logits, active_labels)
        
        total_loss += loss.item()
        loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (global_step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer=optimizer)
            scaler.update()
            optimizer.zero_grad()

            if clear_cache:
                torch.cuda.empty_cache()

        end_time = time.time()
        samples_per_sec = len(batch['input_features']) / (end_time - start_time)

        if global_step % log_interval == 0:
            writer.add_scalar(tag='Loss/train', scalar_value=total_loss / (global_step + 1), global_step=global_step)
            
            lr = optimizer.param_groups[0].get('lr', None)
            if lr is not None:
                writer.add_scalar('LearningRate', scalar_value=lr, global_step=global_step)
            else:
                if not lr_warning_printed:
                    print(f"Warning: Learning rate is None at step {global_step}")
                    lr_warning_printed = True

            writer.add_scalar(tag='SamplesPerSec', scalar_value=samples_per_sec, global_step=global_step)

        if global_step % eval_interval == 0:
            model.eval()
            eval_start_time = time.time()
            eval_loss = 0
            all_predictions = []
            all_labels = []
            batch_count = 0
            total_samples = 0
            
            with torch.no_grad():
                for eval_batch in tqdm(eval_loader, desc=f"Evaluating (Step {global_step})", leave=False):
                    input_features = eval_batch['input_features'].to(device)
                    input_ids = eval_batch['input_ids'].to(device)
                    labels = eval_batch['labels'].long().to(device)
                    
                    batch_size = input_features.size(0)
                    total_samples += batch_size
                    
                    input_features_encoded = model.encoder(input_features)
                    decoder_output = model.decoder(input_ids, input_features_encoded)
                    logits = decoder_output.view(-1, decoder_output.size(-1))
                    loss = loss_fn(logits, labels.view(-1))
                    eval_loss += loss.item()
                    all_predictions.extend(torch.argmax(decoder_output, dim=-1).cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())
                    batch_count += 1

            eval_time = time.time() - eval_start_time
            eval_loss_avg = eval_loss / batch_count if batch_count > 0 else 0
            predictions = {"predictions": np.array(all_predictions, dtype=object), "label_ids": np.array(all_labels, dtype=object)}
            metrics = compute_metrics(pred=predictions, tokenizer=tokenizer)
            
            writer.add_scalar('Loss/eval', eval_loss_avg, global_step)
            writer.add_scalar('WER', metrics['wer'], global_step)
            writer.add_scalar('EvalSamples', total_samples, global_step)
            writer.add_scalar('EvalTimeSeconds', eval_time, global_step)
            
            lr = optimizer.param_groups[0].get('lr', 0)
            
            print("\n" + "="*80)
            print(f"EVALUATION REPORT - STEP {global_step}")
            print("="*80)
            print(f"Metrics:")
            print(f"  • Loss:               {eval_loss_avg:.4f}")
            print(f"  • Word Error Rate:    {metrics['wer']:.2f}%")
            print(f"  • Character Error Rate: {metrics.get('cer', 0):.2f}%")
            print(f"Stats:")
            print(f"  • Learning Rate:      {lr:.8f}")
            print(f"  • Eval Batches:       {batch_count}")
            print(f"  • Eval Samples:       {total_samples}")
            print(f"  • Eval Time:          {eval_time:.2f}s ({total_samples/eval_time:.2f} samples/sec)")
            print(f"  • Training Speed:     {samples_per_sec:.2f} samples/sec")
            
            if len(all_predictions) > 0:
                print("\nSample Predictions:")
                sample_indices = range(min(3, len(all_predictions)))
                for idx in sample_indices:
                    pred_str = tokenizer.decode(all_predictions[idx], skip_special_tokens=True)
                    label_str = tokenizer.decode(all_labels[idx], skip_special_tokens=True)
                    print(f"  Example {idx+1}:")
                    print(f"    • Reference: {label_str}")
                    print(f"    • Prediction: {pred_str}")
            print("="*80 + "\n")
            
            logging.info(f"EVALUATION STEP {global_step} - WER: {metrics['wer']:.2f}%, Loss: {eval_loss_avg:.4f}, LR: {lr:.8f}")
            scheduler.step(eval_loss_avg)
            model.train()

        if global_step % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{global_step}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved at step {global_step} to {checkpoint_path}")
            logging.info(f"Model saved at step {global_step} to {checkpoint_path}")

        if global_step % 5000 == 0 and global_step > 0:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        global_step += 1
        step_in_report += 1
        progress_bar.update(1)
        
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed after {global_step} steps. Final model saved to {final_model_path}")
    writer.close()
    progress_bar.close()


if __name__ == "__main__":

    checkpoint_dir = './output/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join("./output/logs", datetime.now().strftime(format="%m-%d_%H"))
    os.makedirs(name=log_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(log_dir, 'training.log'), filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    token=""

    extractor = WhisperFeatureExtractor.from_pretrained(
        pretrained_model_name_or_path="openai/whisper-small", 
        feature_size=128, sample_rate=16000, do_normalize=True)

    tokenizer = WhisperTokenizerFast.from_pretrained(
        pretrained_model_name_or_path="openai/whisper-small", 
        language="en", task="transcribe")

    processor = WhisperProcessor.from_pretrained(
        pretrained_model_name_or_path="openai/whisper-small", 
        feature_extractor=extractor,
        tokenizer=tokenizer)

    dataset = IterableDatasetDict()

    dataset["train"] = load_dataset(
        path="mozilla-foundation/common_voice_17_0", split="train",
        name="en", streaming=True, token=token, 
        trust_remote_code=True).shuffle()#.take(10000)

    dataset["test"] = load_dataset(
        path="mozilla-foundation/common_voice_17_0",
        name="en", split="test", streaming=True, 
        token=token, trust_remote_code=True).take(500) #type: ignore
    
    dataset = dataset.cast_column(column="audio", feature=Audio(sampling_rate=16000))
    dataset = dataset.map(function=process_fn).with_format(type="torch")                       
    dataset = dataset.select_columns(column_names=["labels", "input_features"])

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, extractor=extractor,
        tokenizer=tokenizer)

    train_dataloader = DataLoader(
        dataset=dataset["train"], 
        batch_size=1, 
        collate_fn=data_collator,
        num_workers=0 )

    eval_dataloader = DataLoader(
        dataset=dataset["test"],
        batch_size=1,
        collate_fn=data_collator,
        num_workers=0 )
    
    param = Dimensions(
        mels=128,
        audio_ctx=1500,
        audio_head=8,
        audio_layerA=8,
        audio_layerB=0,
        audio_state=512,
        vocab=51865,
        text_ctx=448,
        text_head=4,
        text_layerA=4,
        text_layerB=0,
        text_state=512,
        checkpoint=False,
        dropout=0.001,
        activation="gelu",
    )

    model = Echo(param=param).to(device=device)
    model.init_weights()

    optimizer = torch.optim.Adafactor(params=model.parameters(), lr=0.025, 
                         beta2_decay=-0.8, eps=(1e-10, 1e-4), 
                         d=1.0, weight_decay=0.0, 
                         foreach=None, maximize=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                                           last_epoch = -1, T_max=100000, eta_min=0)

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda _: 1.0)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    train_and_evaluate(model=model, 
        tokenizer=tokenizer, 
        train_loader=train_dataloader, 
        eval_loader=eval_dataloader, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        loss_fn=loss_fn, 
        max_steps=100000,
        device='cuda', 
        accumulation_steps=1, 
        clear_cache=True, 
        log_interval=10, 
        eval_interval=1000, 
        save_interval=25000, 
        checkpoint_dir=checkpoint_dir, 
        log_dir=log_dir
        )


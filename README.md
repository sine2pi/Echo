Whisper like ASR transformer model but with some advananced ideas. Experimental.
Internally/dynamically auto tunes itself on window_size for the hybrid attention, base frequency for the embeddings and relative bias for certain tokens during training. The hybrid attention has both global and local attentions with the global informing the local on what it should on. More details here : https://github.com/sine2pi/Focused-Attention


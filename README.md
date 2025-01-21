This model does not include the focused attention blocks. It includes the givens orthogonal rotary for the encoder, dynamic relative positional bias and dynamic base frequency that internally update based on loss, custom multihead, and learned sinusoidal for the decoder.

https://github.com/sine2pi/givens-orthogonal-embeddings

https://github.com/sine2pi/big-multihead-attention

https://github.com/sine2pi/smart-sinusoid

https://colab.research.google.com/drive/1XW6DG-7c9bqhETnmgz29r_-S5Wcjw6uQ?usp=sharing

dataset used: https://huggingface.co/datasets/fixie-ai/librispeech_asr


<img width="850" alt="eval" src="https://github.com/user-attachments/assets/3c165fdb-c12a-4d8f-8189-515927b00be8" />

https://huggingface.co/Sin2pi/Echo2

# Pilot run
Both models were initialized from scratch and trained on the same data using a simple training setup outlined in modelA_with_trainer.ipynb. Hugging face trainer and datasets were used for consistency and reproducibility.  A lot more testing needs to be done but it looks promising 🙂. 
Both models can be considered "medium". 

    #Openai config :
    dims = ModelDimensions(
        bos_token_id=50257,
        decoder_start_token_id=50258,
        eos_token_id=50257,
        init_std=0.02,
        n_audio_ctx=1500,
        n_audio_head=16,
        n_audio_layer=24,
        n_audio_state=1024,
        n_mels=128,
        n_text_ctx=448,
        n_text_head=16,
        n_text_layer=16,
        n_text_state=1024,
        pad_token_id=50257,
        n_vocab=51865,
    )

    #Echo config :
    config = EchoConfig(
        base=10000,
        bos_token_id=50257,
        decoder_start_token_id=50258,
        eos_token_id=50257,
        init_std=0.02,
        max_dist=128,
        n_audio_ctx=1500,
        n_audio_head=16,
        n_audio_layer=20, 
        n_audio_state=1024,
        n_mels=128,
        n_text_ctx=448,
        n_text_head=16,
        n_text_layer=16,
        n_text_state=1024,
        pad_token_id=50257,
        n_vocab=51865,
        )

##  Echo
### Evaluation: - Step 1000 - Loss: 2.7929 - WER - 41.200828 
step-1000
##### Last Prediction:  his all hopes from back who may I will never see serious again unless I nor with me the not so words we will said of us turn back cried his hopes and its not added!!!!!!!!!!!!!!!!!!!!!!!!!!!
##### Label: At all events turn back who may I will never see Greece again unless I carry with me the Golden Fleece we will none of us turn back cried his nine and forty brave comrades

------


## Whisper
### Evaluation: - Step 1000 - Loss: 28.3524 - WER - 78.379178
step-1000
##### Last Prediction: And the all events the to who may had will the the the to to I the a me other to to to to we will to the us to to to his to the to to 
to!!!!!!!!!!!!!!!!!!!!!!!!!!!
##### Label: At all events turn back who may I will never see Greece again unless I carry with me the Golden Fleece we will none of us turn back cried his nine and forty brave comrades

------



### Metrics for an Echo model (medium) from scratch (not pretrained) over 1000 steps of training :

<table>
</div>
    <progress value='1000' max='1000' style='width:300px; height:20px; vertical-align: middle;'></progress>
</div>
<table border="2" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Wer</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>No log</td>
      <td>11.088348</td>
      <td>114.226560</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>50</td>
      <td>8.486400</td>
      <td>8.280888</td>
      <td>108.754806</td>
      <td>0.071915</td>
      <td>0.014912</td>
      <td>0.071915</td>
      <td>0.021198</td>
    </tr>
    <tr>
      <td>100</td>
      <td>7.762600</td>
      <td>7.546603</td>
      <td>102.218279</td>
      <td>0.096644</td>
      <td>0.052328</td>
      <td>0.096644</td>
      <td>0.054700</td>
    </tr>
    <tr>
      <td>150</td>
      <td>7.126700</td>
      <td>7.246966</td>
      <td>100.473233</td>
      <td>0.089579</td>
      <td>0.055702</td>
      <td>0.089579</td>
      <td>0.058204</td>
    </tr>
    <tr>
      <td>200</td>
      <td>7.581300</td>
      <td>7.112317</td>
      <td>105.501331</td>
      <td>0.087560</td>
      <td>0.064674</td>
      <td>0.087560</td>
      <td>0.065928</td>
    </tr>
    <tr>
      <td>250</td>
      <td>7.182800</td>
      <td>6.953177</td>
      <td>102.395741</td>
      <td>0.118849</td>
      <td>0.060984</td>
      <td>0.118849</td>
      <td>0.067212</td>
    </tr>
    <tr>
      <td>300</td>
      <td>7.429000</td>
      <td>6.860844</td>
      <td>87.133984</td>
      <td>0.178148</td>
      <td>0.124331</td>
      <td>0.178148</td>
      <td>0.135332</td>
    </tr>
    <tr>
      <td>350</td>
      <td>6.206500</td>
      <td>6.202377</td>
      <td>91.866312</td>
      <td>0.207923</td>
      <td>0.155767</td>
      <td>0.207923</td>
      <td>0.163332</td>
    </tr>
    <tr>
      <td>400</td>
      <td>5.880600</td>
      <td>5.737767</td>
      <td>84.945282</td>
      <td>0.257128</td>
      <td>0.202511</td>
      <td>0.257128</td>
      <td>0.209760</td>
    </tr>
    <tr>
      <td>450</td>
      <td>5.165900</td>
      <td>5.352962</td>
      <td>82.460810</td>
      <td>0.282614</td>
      <td>0.245499</td>
      <td>0.282614</td>
      <td>0.244798</td>
    </tr>
    <tr>
      <td>500</td>
      <td>4.864500</td>
      <td>5.014288</td>
      <td>78.053830</td>
      <td>0.334847</td>
      <td>0.290442</td>
      <td>0.334847</td>
      <td>0.287975</td>
    </tr>
    <tr>
      <td>550</td>
      <td>4.800300</td>
      <td>4.641736</td>
      <td>72.848270</td>
      <td>0.383043</td>
      <td>0.305948</td>
      <td>0.383043</td>
      <td>0.322512</td>
    </tr>
    <tr>
      <td>600</td>
      <td>4.749200</td>
      <td>4.243003</td>
      <td>64.862467</td>
      <td>0.448902</td>
      <td>0.360028</td>
      <td>0.448902</td>
      <td>0.380816</td>
    </tr>
    <tr>
      <td>650</td>
      <td>3.972400</td>
      <td>3.941687</td>
      <td>60.130139</td>
      <td>0.495079</td>
      <td>0.389808</td>
      <td>0.495079</td>
      <td>0.422348</td>
    </tr>
    <tr>
      <td>700</td>
      <td>3.808100</td>
      <td>3.702391</td>
      <td>58.947057</td>
      <td>0.504668</td>
      <td>0.363262</td>
      <td>0.504668</td>
      <td>0.409206</td>
    </tr>
    <tr>
      <td>750</td>
      <td>3.712500</td>
      <td>3.417185</td>
      <td>53.179533</td>
      <td>0.557406</td>
      <td>0.439535</td>
      <td>0.557406</td>
      <td>0.479163</td>
    </tr>
    <tr>
      <td>800</td>
      <td>3.816800</td>
      <td>3.236211</td>
      <td>51.730257</td>
      <td>0.572294</td>
      <td>0.423974</td>
      <td>0.572294</td>
      <td>0.473410</td>
    </tr>
    <tr>
      <td>850</td>
      <td>3.274800</td>
      <td>3.051460</td>
      <td>46.909198</td>
      <td>0.612415</td>
      <td>0.472236</td>
      <td>0.612415</td>
      <td>0.518319</td>
    </tr>
    <tr>
      <td>900</td>
      <td>2.740500</td>
      <td>2.923042</td>
      <td>44.661343</td>
      <td>0.641686</td>
      <td>0.519547</td>
      <td>0.641686</td>
      <td>0.561009</td>
    </tr>
    <tr>
      <td>950</td>
      <td>2.811200</td>
      <td>2.829455</td>
      <td>40.757172</td>
      <td>0.673732</td>
      <td>0.560039</td>
      <td>0.673732</td>
      <td>0.598650</td>
    </tr>
    <tr>
      <td>1000</td>
      <td>2.752000</td>
      <td>2.792902</td>
      <td>41.200828</td>
      <td>0.669190</td>
      <td>0.543252</td>
      <td>0.669190</td>
      <td>0.586206</td>
    </tr>
  </tbody>
</table><p>





### Metrics for an Openai whisper model (medium) from scratch (not pretrained) over 1000 steps of training :

<table>
</div>
    <progress value='1000' max='1000' style='width:300px; height:20px; vertical-align: middle;'></progress>
</div>
<table border="2" class="dataframe">
 <thead>
   <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Wer</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>20</td>
      <td>94.403200</td>
      <td>94.263489</td>
      <td>91.008577</td>
      <td>0.025738</td>
      <td>0.014635</td>
      <td>0.025738</td>
      <td>0.017516</td>
    </tr>
    <tr>
      <td>40</td>
      <td>85.520300</td>
      <td>77.752335</td>
      <td>80.981958</td>
      <td>0.025486</td>
      <td>0.012869</td>
      <td>0.025486</td>
      <td>0.017075</td>
    </tr>
    <tr>
      <td>60</td>
      <td>78.748400</td>
      <td>72.717720</td>
      <td>83.407276</td>
      <td>0.050467</td>
      <td>0.050467</td>
      <td>0.050467</td>
      <td>0.050467</td>
    </tr>
    <tr>
      <td>80</td>
      <td>71.716200</td>
      <td>63.925533</td>
      <td>82.283348</td>
      <td>0.025486</td>
      <td>0.025486</td>
      <td>0.025486</td>
      <td>0.025486</td>
    </tr>
    <tr>
      <td>100</td>
      <td>61.559900</td>
      <td>64.625153</td>
      <td>101.863354</td>
      <td>0.070654</td>
      <td>0.015002</td>
      <td>0.070654</td>
      <td>0.021354</td>
    </tr>
    <tr>
      <td>120</td>
      <td>57.366200</td>
      <td>57.688732</td>
      <td>90.505768</td>
      <td>0.043654</td>
      <td>0.013775</td>
      <td>0.043654</td>
      <td>0.018800</td>
    </tr>
    <tr>
      <td>140</td>
      <td>54.002500</td>
      <td>51.427681</td>
      <td>84.057971</td>
      <td>0.025233</td>
      <td>0.012617</td>
      <td>0.025233</td>
      <td>0.016822</td>
    </tr>
    <tr>
      <td>160</td>
      <td>53.483600</td>
      <td>48.576084</td>
      <td>78.231293</td>
      <td>0.025233</td>
      <td>0.012617</td>
      <td>0.025233</td>
      <td>0.016822</td>
    </tr>
    <tr>
      <td>180</td>
      <td>47.297000</td>
      <td>47.999882</td>
      <td>88.671991</td>
      <td>0.061065</td>
      <td>0.050653</td>
      <td>0.061065</td>
      <td>0.050832</td>
    </tr>
    <tr>
      <td>200</td>
      <td>49.127000</td>
      <td>46.261742</td>
      <td>85.803017</td>
      <td>0.068887</td>
      <td>0.051025</td>
      <td>0.068887</td>
      <td>0.051551</td>
    </tr>
    <tr>
      <td>220</td>
      <td>42.094100</td>
      <td>43.525475</td>
      <td>75.510204</td>
      <td>0.027252</td>
      <td>0.014096</td>
      <td>0.027252</td>
      <td>0.018415</td>
    </tr>
    <tr>
      <td>240</td>
      <td>49.604300</td>
      <td>42.715565</td>
      <td>83.703046</td>
      <td>0.025233</td>
      <td>0.012617</td>
      <td>0.025233</td>
      <td>0.016822</td>
    </tr>
    <tr>
      <td>260</td>
      <td>40.123900</td>
      <td>50.845093</td>
      <td>117.598344</td>
      <td>0.019934</td>
      <td>0.000403</td>
      <td>0.019934</td>
      <td>0.000790</td>
    </tr>
    <tr>
      <td>280</td>
      <td>43.100900</td>
      <td>40.379704</td>
      <td>80.893227</td>
      <td>0.062074</td>
      <td>0.051841</td>
      <td>0.062074</td>
      <td>0.052830</td>
    </tr>
    <tr>
      <td>300</td>
      <td>46.004100</td>
      <td>39.359657</td>
      <td>86.187518</td>
      <td>0.062831</td>
      <td>0.059012</td>
      <td>0.062831</td>
      <td>0.059481</td>
    </tr>
    <tr>
      <td>320</td>
      <td>40.832000</td>
      <td>39.853710</td>
      <td>93.286010</td>
      <td>0.025233</td>
      <td>0.012617</td>
      <td>0.025233</td>
      <td>0.016822</td>
    </tr>
    <tr>
      <td>340</td>
      <td>43.416100</td>
      <td>39.105030</td>
      <td>98.432416</td>
      <td>0.050214</td>
      <td>0.013454</td>
      <td>0.050214</td>
      <td>0.018442</td>
    </tr>
    <tr>
      <td>360</td>
      <td>37.006500</td>
      <td>37.590755</td>
      <td>97.722567</td>
      <td>0.090336</td>
      <td>0.052648</td>
      <td>0.090336</td>
      <td>0.054604</td>
    </tr>
    <tr>
      <td>380</td>
      <td>41.060600</td>
      <td>37.231003</td>
      <td>96.687371</td>
      <td>0.091597</td>
      <td>0.052742</td>
      <td>0.091597</td>
      <td>0.054778</td>
    </tr>
    <tr>
      <td>400</td>
      <td>39.723700</td>
      <td>37.233490</td>
      <td>98.846495</td>
      <td>0.032046</td>
      <td>0.012809</td>
      <td>0.032046</td>
      <td>0.017152</td>
    </tr>
    <tr>
      <td>420</td>
      <td>36.098600</td>
      <td>34.588448</td>
      <td>80.419994</td>
      <td>0.049962</td>
      <td>0.016635</td>
      <td>0.049962</td>
      <td>0.023593</td>
    </tr>
    <tr>
      <td>440</td>
      <td>36.171700</td>
      <td>35.197464</td>
      <td>84.383319</td>
      <td>0.056523</td>
      <td>0.051583</td>
      <td>0.056523</td>
      <td>0.052213</td>
    </tr>
    <tr>
      <td>460</td>
      <td>40.916200</td>
      <td>34.445030</td>
      <td>91.097308</td>
      <td>0.083018</td>
      <td>0.054857</td>
      <td>0.083018</td>
      <td>0.057037</td>
    </tr>
    <tr>
      <td>480</td>
      <td>31.560100</td>
      <td>34.273525</td>
      <td>93.433895</td>
      <td>0.062579</td>
      <td>0.058669</td>
      <td>0.062579</td>
      <td>0.053839</td>
    </tr>
    <tr>
      <td>500</td>
      <td>31.555100</td>
      <td>35.701134</td>
      <td>93.611358</td>
      <td>0.029018</td>
      <td>0.013088</td>
      <td>0.029018</td>
      <td>0.017486</td>
    </tr>
    <tr>
      <td>520</td>
      <td>31.655800</td>
      <td>33.348915</td>
      <td>96.598639</td>
      <td>0.059046</td>
      <td>0.050900</td>
      <td>0.059046</td>
      <td>0.051067</td>
    </tr>
    <tr>
      <td>540</td>
      <td>31.640500</td>
      <td>35.189606</td>
      <td>105.057675</td>
      <td>0.093111</td>
      <td>0.058158</td>
      <td>0.093111</td>
      <td>0.062370</td>
    </tr>
    <tr>
      <td>560</td>
      <td>33.660200</td>
      <td>32.557995</td>
      <td>83.584738</td>
      <td>0.062579</td>
      <td>0.062538</td>
      <td>0.062579</td>
      <td>0.059354</td>
    </tr>
    <tr>
      <td>580</td>
      <td>32.580900</td>
      <td>32.592445</td>
      <td>94.971902</td>
      <td>0.063084</td>
      <td>0.065103</td>
      <td>0.063084</td>
      <td>0.052611</td>
    </tr>
    <tr>
      <td>600</td>
      <td>35.922800</td>
      <td>34.419098</td>
      <td>106.033718</td>
      <td>0.095887</td>
      <td>0.052758</td>
      <td>0.095887</td>
      <td>0.054830</td>
    </tr>
    <tr>
      <td>620</td>
      <td>30.603100</td>
      <td>31.972431</td>
      <td>95.652174</td>
      <td>0.067626</td>
      <td>0.055185</td>
      <td>0.067626</td>
      <td>0.054872</td>
    </tr>
    <tr>
      <td>640</td>
      <td>30.338900</td>
      <td>31.226700</td>
      <td>82.431233</td>
      <td>0.069140</td>
      <td>0.056380</td>
      <td>0.069140</td>
      <td>0.059379</td>
    </tr>
    <tr>
      <td>660</td>
      <td>28.825100</td>
      <td>30.914654</td>
      <td>85.595978</td>
      <td>0.076962</td>
      <td>0.062543</td>
      <td>0.076962</td>
      <td>0.058539</td>
    </tr>
    <tr>
      <td>680</td>
      <td>29.440200</td>
      <td>30.539631</td>
      <td>86.424135</td>
      <td>0.085037</td>
      <td>0.062852</td>
      <td>0.085037</td>
      <td>0.064045</td>
    </tr>
    <tr>
      <td>700</td>
      <td>29.113400</td>
      <td>30.472658</td>
      <td>83.614315</td>
      <td>0.066616</td>
      <td>0.060417</td>
      <td>0.066616</td>
      <td>0.058387</td>
    </tr>
    <tr>
      <td>720</td>
      <td>31.490900</td>
      <td>30.239878</td>
      <td>90.860692</td>
      <td>0.094373</td>
      <td>0.054398</td>
      <td>0.094373</td>
      <td>0.057342</td>
    </tr>
    <tr>
      <td>740</td>
      <td>28.677400</td>
      <td>29.869360</td>
      <td>82.135463</td>
      <td>0.091597</td>
      <td>0.058511</td>
      <td>0.091597</td>
      <td>0.063113</td>
    </tr>
    <tr>
      <td>760</td>
      <td>34.107900</td>
      <td>29.662447</td>
      <td>83.762201</td>
      <td>0.092607</td>
      <td>0.063801</td>
      <td>0.092607</td>
      <td>0.065499</td>
    </tr>
    <tr>
      <td>780</td>
      <td>34.549900</td>
      <td>29.734320</td>
      <td>89.559302</td>
      <td>0.086551</td>
      <td>0.058246</td>
      <td>0.086551</td>
      <td>0.062451</td>
    </tr>
    <tr>
      <td>800</td>
      <td>30.139900</td>
      <td>29.545433</td>
      <td>87.281869</td>
      <td>0.072925</td>
      <td>0.059602</td>
      <td>0.072925</td>
      <td>0.062393</td>
    </tr>
    <tr>
      <td>820</td>
      <td>31.375100</td>
      <td>29.174932</td>
      <td>85.595978</td>
      <td>0.092102</td>
      <td>0.055116</td>
      <td>0.092102</td>
      <td>0.058551</td>
    </tr>
    <tr>
      <td>840</td>
      <td>27.732400</td>
      <td>28.951723</td>
      <td>78.970719</td>
      <td>0.081756</td>
      <td>0.064148</td>
      <td>0.081756</td>
      <td>0.065059</td>
    </tr>
    <tr>
      <td>860</td>
      <td>30.152400</td>
      <td>28.902292</td>
      <td>87.015676</td>
      <td>0.076205</td>
      <td>0.064815</td>
      <td>0.076205</td>
      <td>0.064552</td>
    </tr>
    <tr>
      <td>880</td>
      <td>34.486900</td>
      <td>28.868065</td>
      <td>91.688849</td>
      <td>0.071411</td>
      <td>0.069928</td>
      <td>0.071411</td>
      <td>0.065083</td>
    </tr>
    <tr>
      <td>900</td>
      <td>25.140400</td>
      <td>28.629511</td>
      <td>80.005915</td>
      <td>0.085289</td>
      <td>0.065063</td>
      <td>0.085289</td>
      <td>0.065973</td>
    </tr>
    <tr>
      <td>920</td>
      <td>31.664500</td>
      <td>28.500097</td>
      <td>77.846791</td>
      <td>0.090336</td>
      <td>0.065317</td>
      <td>0.090336</td>
      <td>0.070051</td>
    </tr>
    <tr>
      <td>940</td>
      <td>26.276000</td>
      <td>28.503267</td>
      <td>81.514345</td>
      <td>0.084532</td>
      <td>0.068520</td>
      <td>0.084532</td>
      <td>0.068082</td>
    </tr>
    <tr>
      <td>960</td>
      <td>30.347800</td>
      <td>28.445673</td>
      <td>80.390417</td>
      <td>0.078981</td>
      <td>0.074482</td>
      <td>0.078981</td>
      <td>0.069878</td>
    </tr>
    <tr>
      <td>980</td>
      <td>25.139200</td>
      <td>28.366508</td>
      <td>79.177758</td>
      <td>0.092354</td>
      <td>0.064299</td>
      <td>0.092354</td>
      <td>0.062803</td>
    </tr>
    <tr>
      <td>1000</td>
      <td>28.350800</td>
      <td>28.352432</td>
      <td>78.379178</td>
      <td>0.089831</td>
      <td>0.064653</td>
      <td>0.089831</td>
      <td>0.067847</td>
    </tr>
  </tbody>
</table><p>


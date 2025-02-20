# Models

All our models are released under a research-only [RAIL Model License](LICENSE_MODEL).

## Downloading the pretrained models

All our models described in our tech report are released on our GitHub.

We use the following script to convert to tar files:

```bash
# Performs something akin to:
# tar czf - filename | split -b 1900MB - filename.tar.gz.part_
python scripts/handle_checkpoints.py \
--mode create \
--checkpoint_dir XXXX \
--outdir vavam_release \
--maxsize 1900MB
```

If a model is chunked into several tar files (VaViM-L and VaVAM-L), you can merge them using the following command:

1. Download all tar files.
2. Put them in a single folder (e.g. `vavam_release_chunks`).
3. Run the following command:

```bash
# Performs something akin to:
# cat filename.tar.gz.part_* > filename.tar.gz
# tar xzf filename.tar.gz
python scripts/handle_checkpoints.py \
--mode extract \
--checkpoint_dir vavam_release_chunks \
--outdir vavam_release
```

## Available models

### Main models

Here are the links for our main VaViM and VaVAM models:

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># of<br />params</th>
      <th>VaViM</th>
      <th>VaVAM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VaVAM-S</td>
      <td align="right">185M + 21M</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_768_pretrained_139k_total_155k.tar.gz">part 1</a></td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_768_pretrained_139k.tar.gz">part 1</a></td>
    </tr>
    <tr>
      <td>VaVAM-B</td>
      <td align="right">318M + 38M</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_1024_pretrained_139k_total_155k.tar.gz">part 1</a></td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_1024_pretrained_139k.tar.gz">part 1</a></td>
    </tr>
    <tr>
      <td>VaVAM-L</td>
      <td align="right">1.2B + 150M</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_2048_pretrained_139k_total_155k_chunked.tar.gz.part_aa">part 1</a>, <a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_2048_pretrained_139k_total_155k_chunked.tar.gz.part_ab">part 2</a>, <a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_2048_pretrained_139k_total_155k_chunked.tar.gz.part_ac">part 3</a></td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_2048_pretrained_139k_chunked.tar.gz.part_aa">part 1</a>, <a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_2048_pretrained_139k_chunked.tar.gz.part_ab">part 2</a>, <a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_2048_pretrained_139k_chunked.tar.gz.part_ac">part 3</a>, <a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_2048_pretrained_139k_chunked.tar.gz.part_ad">part 4</a></td>
    </tr>
  </tbody>
</table>

### VaViM only

We also release the different checkpoints that helped up compute our scaling laws. Here are the different VaViM models, with different sizes and trained on different amounts of data:

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># params<br />(in M)</th>
      <th># data<br />(in ×10<sup>3</sup>)</th>
      <th>pre-trained</th>
      <th>fine-tuned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VaViM-S</td>
      <td align="right">185</td>
      <td align="right">38</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_768_pretrained_38k.tar.gz">part 1</a></td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_768_pretrained_38k_total_54k.tar.gz">part 1</a></td>
    </tr>
    <tr>
      <td>VaViM-S</td>
      <td align="right">185</td>
      <td align="right">77</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_768_pretrained_77k.tar.gz">part 1</a></td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_768_pretrained_77k_total_93k.tar.gz">part 1</a></td>
    </tr>
    <tr>
      <td>VaViM-S</td>
      <td align="right">185</td>
      <td align="right">116</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_768_pretrained_116k.tar.gz">part 1</a></td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_768_pretrained_116k_total_132k.tar.gz">part 1</a></td>
    </tr>
    <tr>
      <td>VaViM-S</td>
      <td align="right">185</td>
      <td align="right">139</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_768_pretrained_139k.tar.gz">part 1</a></td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_768_pretrained_139k_total_155k.tar.gz">part 1</a></td>
    </tr>
    <tr>
      <td>VaViM-B</td>
      <td align="right">318</td>
      <td align="right">38</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_1024_pretrained_38k.tar.gz">part 1</a></td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_1024_pretrained_38k_total_54k.tar.gz">part 1</a></td>
    </tr>
    <tr>
      <td>VaViM-B</td>
      <td align="right">318</td>
      <td align="right">77</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_1024_pretrained_77k.tar.gz">part 1</a></td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_1024_pretrained_77k_total_93k.tar.gz">part 1</a></td>
    </tr>
    <tr>
      <td>VaViM-B</td>
      <td align="right">318</td>
      <td align="right">116</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_1024_pretrained_116k.tar.gz">part 1</a></td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_1024_pretrained_116k_total_132k.tar.gz">part 1</a></td>
    </tr>
    <tr>
      <td>VaViM-B</td>
      <td align="right">318</td>
      <td align="right">139</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_1024_pretrained_139k.tar.gz">part 1</a></td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_1024_pretrained_139k_total_155k.tar.gz">part 1</a></td>
    </tr>
    <tr>
      <td>VaViM-L</td>
      <td align="right">1200</td>
      <td align="right">139</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_2048_pretrained_139k_chunked.tar.gz.part_aa">part 1</a>, <a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_2048_pretrained_139k_chunked.tar.gz.part_ab">part 2</a>, <a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_2048_pretrained_139k_chunked.tar.gz.part_ac">part 3</a></td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_2048_pretrained_139k_total_155k_chunked.tar.gz.part_aa">part 1</a>, <a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_2048_pretrained_139k_total_155k_chunked.tar.gz.part_ab">part 2</a>, <a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/width_2048_pretrained_139k_total_155k_chunked.tar.gz.part_ac">part 2</a></td>
    </tr>
  </tbody>
</table>

### VaVAM

We trained VaVAM models given the VaViM models. Here are the different VaVAM models, with their corresponding amount of pre-training data:

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># params</th>
      <th># data<br />(in ×10<sup>3</sup>)</th>
      <th>VaVAM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VaVAM-S</td>
      <td align="right">185M + 21M</td>
      <td align="right">38</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_768_pretrained_38k.tar.gz">part 1</a></td>
    </tr>
    <tr>
      <td>VaVAM-S</td>
      <td align="right">185M + 21M</td>
      <td align="right">77</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_1024_pretrained_77k.tar.gz">part 1</a></td>
    </tr>
    <tr>
      <td>VaVAM-S</td>
      <td align="right">185M + 21M</td>
      <td align="right">116</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_768_pretrained_116k.tar.gz">part 1</a></td>
    </tr>
    <tr>
      <td>VaVAM-S</td>
      <td align="right">185M + 21M</td>
      <td align="right">139</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_768_pretrained_139k.tar.gz">part 1</a></td>
    </tr>
    <tr>
      <td>VaVAM-B</td>
      <td align="right">318M + 38M</td>
      <td align="right">38</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_1024_pretrained_38k.tar.gz">part 1</a></td>
    </tr>
    <tr>
      <td>VaVAM-B</td>
      <td align="right">318M + 38M</td>
      <td align="right">77</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_1024_pretrained_77k.tar.gz">part 1</a></td>
    </tr>
    <tr>
      <td>VaVAM-B</td>
      <td align="right">318M + 38M</td>
      <td align="right">116</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_1024_pretrained_116k.tar.gz">part 1</a></td>
    </tr>
    <tr>
      <td>VaVAM-B</td>
      <td align="right">318M + 38M</td>
      <td align="right">139</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_1024_pretrained_139k.tar.gz">part 1</a></td>
    </tr>
    <tr>
      <td>VaVAM-L</td>
      <td align="right">1.2B + 150M</td>
      <td align="right">139</td>
      <td><a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_2048_pretrained_139k_chunked.tar.gz.part_aa">part 1</a>, <a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_2048_pretrained_139k_chunked.tar.gz.part_ab">part 2</a>, <a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_2048_pretrained_139k_chunked.tar.gz.part_ac">part 3</a>, <a href="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_2048_pretrained_139k_chunked.tar.gz.part_ad">part 4</a></td>
    </tr>
  </tbody>
</table>

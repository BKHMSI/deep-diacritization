# Deep Diacritization: Efficient Hierarchical Recurrence for Improved Arabic Diacritization
We propose a novel architecture for labelling character sequences that achieves state-of-the-art results on the Tashkeela Arabic diacritization benchmark. The core is a two-level recurrence hierarchy that operates on the word and character levels separately---enabling faster training and inference than comparable traditional models. A cross-level attention module further connects the two, and opens the door for network interpretability. The task module is a softmax classifier that enumerates valid combinations of diacritics. This architecture can be extended with a recurrent decoder that optionally accepts priors from partially diacritized text, improving performance significantly. We employ extra tricks such as sentence dropout and majority voting to further boost the final result. Our best model achieves a WER of 5.34\%, outperforming the previous state-of-the-art with a 30.56\% relative error reduction.

## Results on the Tashkeela Benchmark


<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" rowspan="2">DER/WER</th>
    <th class="tg-c3ow" colspan="2">Including 'no diacritic'</th>
    <th class="tg-c3ow" colspan="2">Excluding 'no diacritic'</th>
  </tr>
  <tr>
    <td class="tg-c3ow">w/ case ending</td>
    <td class="tg-c3ow">w/o case ending</td>
    <td class="tg-c3ow">w/ case ending</td>
    <td class="tg-c3ow">w/o case ending</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky"><a href="https://github.com/Barqawiz/Shakkala">Barqawi, 2017</a></td>
    <td class="tg-c3ow">3.73% / 11.19%</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">2.88% / 6.53%</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">4.36% / 10.89%</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">3.33% / 6.37%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://www.aclweb.org/anthology/D19-5229/">Fadel et al., 2019</a></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">2.60% / 7.69%</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">2.11% / 4.57%</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">3.00% / 7.39%</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">2.42% / 4.44%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://www.researchgate.net/publication/340574877_Multi-components_System_for_Automatic_Arabic_Diacritization">Abbad and Xiong, 2020</a></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">3.39% / 9.94%</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">2.61% / 5.83%</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">3.34% / 7.98%</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">2.43% / 3.98%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">D2 (Ours)</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">1.85% / 5.53%</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">1.49% / 3.27%</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">2.11% / 5.26%</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">1.71% / 3.15%</span></td>
  </tr>
  <tr style="font-weight:bold">
    <td class="tg-0pky"><strong>D3 (Ours)</strong></td>
    <td class="tg-7btt"><strong style="font-style:normal">1.83% / 5.34%</strong></td>
    <td class="tg-7btt"><strong style="font-style:normal">1.48% / 3.11%</strong></td>
    <td class="tg-7btt"><strong style="font-style:normal">2.09% / 5.08%</strong></td>
    <td class="tg-7btt"><strong style="font-style:normal">1.69% / 3.00%</strong></td>
  </tr>
</tbody>
</table>

## Getting Started

## Cite

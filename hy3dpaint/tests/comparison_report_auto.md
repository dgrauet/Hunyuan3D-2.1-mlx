# MLX vs PyTorch Numerical Comparison Report

- latent size: 8x8
- in_channels: 12
- tolerance: 0.01

## Layer-by-layer divergence

| layer | status | max_abs | max_rel | mean_abs | pt_norm | mlx_norm |
| --- | --- | --- | --- | --- | --- | --- |
| conv_in | MATCH | 1.490e-07 | 8.340e-04 | 1.463e-08 | 12.381 | 12.381 |
| down_block_0 | MATCH | 1.579e-04 | 3.697e-02 | 3.160e-05 | 118.044 | 118.044 |
| mid_block | MATCH | 3.700e-04 | 4.228e-03 | 4.825e-05 | 116.926 | 116.926 |
| up_block_0 | MATCH | 1.853e-04 | 3.205e-01 | 1.807e-05 | 106.271 | 106.271 |
| conv_out | MATCH | 1.156e-05 | 1.708e-03 | 3.090e-06 | 2.325 | 2.325 |
| final | MATCH | 1.156e-05 | 1.708e-03 | 3.090e-06 | 2.325 | 2.325 |

**First divergent layer:** none — all layers MATCH

## All comparisons

| comparison | status | max_abs | max_rel |
| --- | --- | --- | --- |
| W:conv_in.weight | MATCH | 0.000e+00 | 0.000e+00 |
| W:conv_in.bias | MATCH | 0.000e+00 | 0.000e+00 |
| W:time_embedding.linear_1.weight | MATCH | 0.000e+00 | 0.000e+00 |
| W:time_embedding.linear_1.bias | MATCH | 0.000e+00 | 0.000e+00 |
| W:time_embedding.linear_2.weight | MATCH | 0.000e+00 | 0.000e+00 |
| W:conv_out.weight | MATCH | 0.000e+00 | 0.000e+00 |
| W:conv_out.bias | MATCH | 0.000e+00 | 0.000e+00 |
| W:conv_norm_out.weight | MATCH | 0.000e+00 | 0.000e+00 |
| W:down_blocks.0.resnets.0.norm1.weight | MATCH | 0.000e+00 | 0.000e+00 |
| W:down_blocks.0.resnets.0.conv1.weight | MATCH | 0.000e+00 | 0.000e+00 |
| W:down_blocks.0.resnets.0.time_emb_proj.weight | MATCH | 0.000e+00 | 0.000e+00 |
| W:down_blocks.0.attentions.0.norm.weight | MATCH | 0.000e+00 | 0.000e+00 |
| W:down_blocks.0.attentions.0.proj_in.weight | MATCH | 0.000e+00 | 0.000e+00 |
| W:down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight | MATCH | 0.000e+00 | 0.000e+00 |
| W:down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.weight | MATCH | 0.000e+00 | 0.000e+00 |
| W:down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.weight | MATCH | 0.000e+00 | 0.000e+00 |
| W:mid_block.attentions.0.transformer_blocks.0.attn1.to_q.weight | MATCH | 0.000e+00 | 0.000e+00 |
| timestep_sinusoidal | MATCH | 3.052e-05 | 1.616e-02 |
| conv_in | MATCH | 1.490e-07 | 8.340e-04 |
| down_block_0 | MATCH | 1.579e-04 | 3.697e-02 |
| mid_block | MATCH | 3.700e-04 | 4.228e-03 |
| up_block_0 | MATCH | 1.853e-04 | 3.205e-01 |
| conv_out | MATCH | 1.156e-05 | 1.708e-03 |
| final | MATCH | 1.156e-05 | 1.708e-03 |
| conv_in PT-vs-MLX(loaded) | MATCH | 1.490e-07 | 8.340e-04 |
| conv_in PT-vs-MLX(direct weight set) | MATCH | 1.490e-07 | 8.340e-04 |
| block0.resnet0 | MATCH | 1.144e-05 | 1.348e-01 |
| block0.attn0 | MATCH | 6.783e-05 | 4.811e-01 |
| block0.resnet1 | MATCH | 7.337e-05 | 5.884e-01 |
| block0.attn1 | MATCH | 8.994e-05 | 4.491e-01 |
| block0.resnet0 (GN fix) | MATCH | 1.144e-05 | 1.348e-01 |
| block0.attn0 (GN fix) | MATCH | 6.783e-05 | 4.811e-01 |
| block0.resnet1 (GN fix) | MATCH | 7.337e-05 | 5.884e-01 |
| block0.attn1 (GN fix) | MATCH | 8.994e-05 | 4.491e-01 |
| block0.resnet0 (PT temb) | MISMATCH | 1.848e+00 | 1.356e+05 |
| block0.attn0 (PT temb) | MISMATCH | 2.204e+00 | 8.776e+04 |
| block0.resnet1 (PT temb) | MISMATCH | 2.461e+00 | 4.740e+03 |
| block0.attn1 (PT temb) | MISMATCH | 2.523e+00 | 8.196e+03 |

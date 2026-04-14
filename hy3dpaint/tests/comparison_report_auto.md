# MLX vs PyTorch Numerical Comparison Report

- latent size: 8x8
- in_channels: 12
- tolerance: 0.01

## Layer-by-layer divergence

| layer | status | max_abs | max_rel | mean_abs | pt_norm | mlx_norm |
| --- | --- | --- | --- | --- | --- | --- |
| conv_in | MATCH | 1.490e-07 | 8.340e-04 | 1.463e-08 | 12.381 | 12.381 |
| down_block_0 | MISMATCH | 1.869e+00 | 3.744e+02 | 2.905e-01 | 118.044 | 117.124 |
| mid_block | MISMATCH | 3.045e+00 | 9.712e+01 | 6.139e-01 | 116.926 | 118.351 |
| up_block_0 | MISMATCH | 1.982e+00 | 3.433e+04 | 2.043e-01 | 106.271 | 108.840 |
| conv_out | MISMATCH | 2.148e-01 | 6.528e+01 | 4.955e-02 | 2.325 | 2.495 |
| final | MISMATCH | 2.148e-01 | 6.528e+01 | 4.955e-02 | 2.325 | 2.495 |

**First divergent layer:** down_block_0

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
| down_block_0 | MISMATCH | 1.869e+00 | 3.744e+02 |
| mid_block | MISMATCH | 3.045e+00 | 9.712e+01 |
| up_block_0 | MISMATCH | 1.982e+00 | 3.433e+04 |
| conv_out | MISMATCH | 2.148e-01 | 6.528e+01 |
| final | MISMATCH | 2.148e-01 | 6.528e+01 |
| conv_in PT-vs-MLX(loaded) | MATCH | 1.490e-07 | 8.340e-04 |
| conv_in PT-vs-MLX(direct weight set) | MATCH | 1.490e-07 | 8.340e-04 |
| block0.resnet0 | MATCH | 1.144e-05 | 1.348e-01 |
| block0.attn0 | MISMATCH | 7.484e-01 | 7.405e+03 |
| block0.resnet1 | MISMATCH | 7.871e-01 | 4.057e+03 |
| block0.attn1 | MISMATCH | 1.021e+00 | 7.713e+03 |
| block0.resnet0 (GN fix) | MATCH | 1.144e-05 | 1.348e-01 |
| block0.attn0 (GN fix) | MISMATCH | 7.484e-01 | 7.405e+03 |
| block0.resnet1 (GN fix) | MISMATCH | 7.871e-01 | 4.057e+03 |
| block0.attn1 (GN fix) | MISMATCH | 1.021e+00 | 7.713e+03 |
| block0.resnet0 (PT temb) | MISMATCH | 1.848e+00 | 1.356e+05 |
| block0.attn0 (PT temb) | MISMATCH | 2.312e+00 | 1.356e+04 |
| block0.resnet1 (PT temb) | MISMATCH | 2.493e+00 | 5.032e+03 |
| block0.attn1 (PT temb) | MISMATCH | 2.680e+00 | 7.389e+03 |

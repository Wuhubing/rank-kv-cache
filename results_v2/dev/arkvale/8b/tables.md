
### stream `plain`, scope `all` — G (miss rate %)

| policy | B=72 | B=80 | B=96 | B=128 | B=256 |
|---|---|---|---|---|---|
| lru | 0.000 (9.28) | 0.000 (6.97) | 0.000 (4.58) | 0.000 (2.61) | 0.000 (0.94) |
| score | 0.226 (8.79) | 0.233 (6.49) | 0.221 [0.215, 0.227] (4.23) | 0.172 [0.167, 0.179] (2.44) | 0.049 (0.93) |
| ema_z(lodo) | 0.258 (8.72) | 0.270 (6.42) | 0.272 [0.268, 0.276] (4.14) | 0.256 [0.250, 0.262] (2.36) | 0.232 (0.89) |
| ema_allrank(lodo) | 0.254 (8.73) | 0.263 (6.43) | 0.263 [0.257, 0.267] (4.16) | 0.241 [0.235, 0.248] (2.37) | 0.207 (0.89) |
| powerlaw(lodo) | 0.283 (8.67) | 0.288 (6.38) | 0.311 [0.307, 0.315] (4.08) | 0.285 [0.281, 0.289] (2.33) | 0.226 (0.89) |
| arkvale_simple | 0.120 (9.02) | 0.175 (6.61) | 0.193 [0.186, 0.200] (4.27) | 0.162 [0.155, 0.170] (2.45) | 0.050 (0.93) |
| ark:quest:1 | 0.226 (8.79) | 0.233 (6.49) | 0.221 [0.215, 0.227] (4.23) | 0.172 [0.167, 0.179] (2.44) | 0.049 (0.93) |
| ark:quest:0 | 0.226 (8.79) | 0.233 (6.49) | 0.221 [0.215, 0.227] (4.23) | 0.172 [0.167, 0.179] (2.44) | 0.049 (0.93) |
| ark:cuboid_mean:1 | -0.365 (10.06) | -0.424 (7.85) | -0.297 [-0.321, -0.272] (5.06) | -0.105 [-0.125, -0.082] (2.72) | -0.012 (0.95) |
| ark:cuboid_mean:0 | -0.365 (10.06) | -0.424 (7.85) | -0.297 [-0.321, -0.272] (5.06) | -0.105 [-0.125, -0.082] (2.72) | -0.012 (0.95) |
| ark:digest(lodo):1 | 0.226 (8.79) | 0.233 (6.49) | 0.221 [0.215, 0.227] (4.23) | 0.172 [0.167, 0.179] (2.44) | 0.049 (0.93) |
| belady | 1.000 (7.13) | 1.000 (4.91) | 1.000 (2.98) | 1.000 (1.61) | 1.000 (0.69) |

### stream `page`, scope `all` — G (miss rate %)

| policy | B=72 | B=80 | B=96 | B=128 | B=256 |
|---|---|---|---|---|---|
| lru | 0.000 (9.30) | 0.000 (6.97) | 0.000 (4.57) | 0.000 (2.61) | 0.000 (0.94) |
| score | 0.227 (8.82) | 0.233 (6.49) | 0.222 [0.216, 0.228] (4.21) | 0.173 [0.167, 0.180] (2.43) | 0.050 (0.93) |
| ema_z(lodo) | 0.259 (8.75) | 0.271 (6.41) | 0.272 [0.268, 0.276] (4.13) | 0.257 [0.251, 0.262] (2.35) | 0.232 (0.88) |
| ema_allrank(lodo) | 0.254 (8.76) | 0.264 (6.43) | 0.263 [0.258, 0.268] (4.15) | 0.242 [0.236, 0.248] (2.36) | 0.207 (0.89) |
| powerlaw(lodo) | 0.283 (8.70) | 0.288 (6.38) | 0.311 [0.307, 0.315] (4.07) | 0.285 [0.281, 0.289] (2.32) | 0.226 (0.88) |
| arkvale_simple | 0.133 (9.02) | 0.179 (6.60) | 0.194 [0.186, 0.201] (4.26) | 0.162 [0.155, 0.170] (2.44) | 0.050 (0.93) |
| ark:quest:1 | 0.187 (8.90) | 0.191 (6.58) | 0.167 [0.163, 0.172] (4.30) | 0.091 [0.084, 0.098] (2.52) | -0.138 (0.98) |
| ark:quest:0 | 0.227 (8.82) | 0.233 (6.49) | 0.222 [0.216, 0.228] (4.21) | 0.173 [0.167, 0.180] (2.43) | 0.050 (0.93) |
| ark:cuboid_mean:1 | -0.400 (10.16) | -0.465 (7.93) | -0.351 [-0.376, -0.326] (5.13) | -0.187 [-0.207, -0.164] (2.79) | -0.201 (0.99) |
| ark:cuboid_mean:0 | -0.362 (10.08) | -0.426 (7.85) | -0.300 [-0.324, -0.275] (5.05) | -0.105 [-0.125, -0.083] (2.71) | -0.011 (0.94) |
| ark:digest(lodo):1 | 0.187 (8.90) | 0.191 (6.58) | 0.167 [0.163, 0.172] (4.30) | 0.091 [0.084, 0.098] (2.52) | -0.138 (0.98) |
| belady | 1.000 (7.16) | 1.000 (4.91) | 1.000 (2.97) | 1.000 (1.60) | 1.000 (0.69) |

### stream `token`, scope `all` — G (miss rate %)

| policy | B=72 | B=80 | B=96 | B=128 | B=256 |
|---|---|---|---|---|---|
| lru | 0.000 (9.50) | 0.000 (7.05) | 0.000 (4.57) | 0.000 (2.59) | 0.000 (0.93) |
| score | 0.225 (9.04) | 0.235 (6.57) | 0.225 [0.218, 0.230] (4.22) | 0.177 [0.171, 0.184] (2.41) | 0.053 (0.92) |
| ema_z(lodo) | 0.255 (8.98) | 0.270 (6.50) | 0.273 [0.269, 0.277] (4.14) | 0.258 [0.253, 0.263] (2.33) | 0.232 (0.87) |
| ema_allrank(lodo) | 0.250 (8.99) | 0.264 (6.51) | 0.265 [0.260, 0.270] (4.15) | 0.244 [0.238, 0.249] (2.35) | 0.208 (0.88) |
| powerlaw(lodo) | 0.281 (8.93) | 0.288 (6.46) | 0.312 [0.307, 0.316] (4.08) | 0.286 [0.282, 0.290] (2.30) | 0.227 (0.87) |
| arkvale_simple | 0.225 (9.04) | 0.235 (6.57) | 0.225 [0.218, 0.230] (4.22) | 0.177 [0.171, 0.184] (2.41) | 0.053 (0.92) |
| ark:quest:1 | 0.186 (9.12) | 0.195 (6.66) | 0.173 [0.169, 0.178] (4.30) | 0.098 [0.092, 0.104] (2.49) | -0.132 (0.96) |
| ark:quest:0 | 0.225 (9.04) | 0.235 (6.57) | 0.225 [0.218, 0.230] (4.22) | 0.177 [0.171, 0.184] (2.41) | 0.053 (0.92) |
| ark:cuboid_mean:1 | -0.376 (10.27) | -0.459 (7.99) | -0.354 [-0.378, -0.330] (5.14) | -0.184 [-0.204, -0.163] (2.77) | -0.197 (0.98) |
| ark:cuboid_mean:0 | -0.341 (10.20) | -0.426 (7.92) | -0.309 [-0.333, -0.284] (5.06) | -0.108 [-0.128, -0.086] (2.70) | -0.010 (0.93) |
| ark:digest(lodo):1 | 0.186 (9.12) | 0.195 (6.66) | 0.173 [0.169, 0.178] (4.30) | 0.098 [0.092, 0.104] (2.49) | -0.132 (0.96) |
| belady | 1.000 (7.46) | 1.000 (5.01) | 1.000 (2.99) | 1.000 (1.59) | 1.000 (0.68) |

### stream `plain`, scope `a` — G (miss rate %)

| policy | B=72 | B=80 | B=96 | B=128 | B=256 |
|---|---|---|---|---|---|
| lru | 0.000 (9.52) | 0.000 (7.22) | 0.000 (4.75) | 0.000 (2.71) | 0.000 (0.97) |
| score | 0.229 (9.02) | 0.239 (6.71) | 0.221 [0.214, 0.226] (4.39) | 0.169 [0.163, 0.177] (2.53) | 0.046 (0.96) |
| ema_z(lodo) | 0.259 (8.96) | 0.273 (6.64) | 0.271 [0.266, 0.275] (4.30) | 0.254 [0.248, 0.260] (2.44) | 0.229 (0.91) |
| ema_allrank(lodo) | 0.254 (8.97) | 0.266 (6.65) | 0.262 [0.256, 0.266] (4.32) | 0.239 [0.233, 0.245] (2.46) | 0.204 (0.92) |
| powerlaw(lodo) | 0.284 (8.90) | 0.291 (6.60) | 0.310 [0.306, 0.314] (4.24) | 0.283 [0.279, 0.287] (2.41) | 0.223 (0.91) |
| arkvale_simple | 0.122 (9.25) | 0.180 (6.83) | 0.192 [0.185, 0.200] (4.43) | 0.158 [0.151, 0.167] (2.54) | 0.047 (0.96) |
| ark:quest:1 | 0.229 (9.02) | 0.239 (6.71) | 0.221 [0.214, 0.226] (4.39) | 0.169 [0.163, 0.177] (2.53) | 0.046 (0.96) |
| ark:quest:0 | 0.229 (9.02) | 0.239 (6.71) | 0.221 [0.214, 0.226] (4.39) | 0.169 [0.163, 0.177] (2.53) | 0.046 (0.96) |
| ark:cuboid_mean:1 | -0.350 (10.28) | -0.401 (8.07) | -0.293 [-0.317, -0.269] (5.24) | -0.108 [-0.128, -0.086] (2.82) | -0.015 (0.98) |
| ark:cuboid_mean:0 | -0.350 (10.28) | -0.401 (8.07) | -0.293 [-0.317, -0.269] (5.24) | -0.108 [-0.128, -0.086] (2.82) | -0.015 (0.98) |
| ark:digest(lodo):1 | 0.229 (9.02) | 0.239 (6.71) | 0.221 [0.214, 0.226] (4.39) | 0.169 [0.163, 0.177] (2.53) | 0.046 (0.96) |
| belady | 1.000 (7.34) | 1.000 (5.09) | 1.000 (3.09) | 1.000 (1.66) | 1.000 (0.71) |

### stream `page`, scope `a` — G (miss rate %)

| policy | B=72 | B=80 | B=96 | B=128 | B=256 |
|---|---|---|---|---|---|
| lru | 0.000 (9.53) | 0.000 (7.21) | 0.000 (4.74) | 0.000 (2.70) | 0.000 (0.97) |
| score | 0.230 (9.03) | 0.240 (6.70) | 0.222 [0.215, 0.228] (4.37) | 0.170 [0.164, 0.178] (2.53) | 0.046 (0.96) |
| ema_z(lodo) | 0.258 (8.96) | 0.273 (6.63) | 0.270 [0.266, 0.274] (4.29) | 0.254 [0.248, 0.260] (2.44) | 0.230 (0.91) |
| ema_allrank(lodo) | 0.254 (8.97) | 0.267 (6.64) | 0.262 [0.257, 0.267] (4.31) | 0.239 [0.233, 0.246] (2.45) | 0.204 (0.91) |
| powerlaw(lodo) | 0.284 (8.91) | 0.292 (6.59) | 0.310 [0.306, 0.315] (4.23) | 0.283 [0.279, 0.288] (2.41) | 0.224 (0.91) |
| arkvale_simple | 0.131 (9.24) | 0.184 (6.82) | 0.193 [0.185, 0.200] (4.42) | 0.158 [0.151, 0.167] (2.54) | 0.047 (0.96) |
| ark:quest:1 | 0.190 (9.11) | 0.198 (6.79) | 0.170 [0.165, 0.174] (4.46) | 0.091 [0.084, 0.098] (2.61) | -0.134 (1.00) |
| ark:quest:0 | 0.230 (9.03) | 0.240 (6.70) | 0.222 [0.215, 0.228] (4.37) | 0.170 [0.164, 0.178] (2.53) | 0.046 (0.96) |
| ark:cuboid_mean:1 | -0.387 (10.36) | -0.440 (8.14) | -0.344 [-0.368, -0.319] (5.31) | -0.186 [-0.206, -0.164] (2.90) | -0.197 (1.02) |
| ark:cuboid_mean:0 | -0.349 (10.28) | -0.403 (8.06) | -0.294 [-0.318, -0.270] (5.23) | -0.108 [-0.128, -0.086] (2.82) | -0.015 (0.97) |
| ark:digest(lodo):1 | 0.190 (9.11) | 0.198 (6.79) | 0.170 [0.165, 0.174] (4.46) | 0.091 [0.084, 0.098] (2.61) | -0.134 (1.00) |
| belady | 1.000 (7.35) | 1.000 (5.09) | 1.000 (3.08) | 1.000 (1.66) | 1.000 (0.70) |

### stream `token`, scope `a` — G (miss rate %)

| policy | B=72 | B=80 | B=96 | B=128 | B=256 |
|---|---|---|---|---|---|
| lru | 0.000 (9.72) | 0.000 (7.29) | 0.000 (4.75) | 0.000 (2.68) | 0.000 (0.96) |
| score | 0.226 (9.25) | 0.241 (6.78) | 0.225 [0.218, 0.231] (4.38) | 0.174 [0.168, 0.182] (2.50) | 0.049 (0.94) |
| ema_z(lodo) | 0.254 (9.19) | 0.273 (6.71) | 0.273 [0.268, 0.276] (4.30) | 0.256 [0.250, 0.261] (2.42) | 0.230 (0.90) |
| ema_allrank(lodo) | 0.250 (9.20) | 0.267 (6.73) | 0.265 [0.259, 0.269] (4.31) | 0.241 [0.235, 0.247] (2.43) | 0.205 (0.90) |
| powerlaw(lodo) | 0.281 (9.13) | 0.292 (6.67) | 0.312 [0.307, 0.316] (4.23) | 0.285 [0.280, 0.289] (2.39) | 0.224 (0.90) |
| arkvale_simple | 0.226 (9.25) | 0.241 (6.78) | 0.225 [0.218, 0.231] (4.38) | 0.174 [0.168, 0.182] (2.50) | 0.049 (0.94) |
| ark:quest:1 | 0.188 (9.32) | 0.202 (6.86) | 0.176 [0.171, 0.180] (4.46) | 0.099 [0.092, 0.105] (2.58) | -0.128 (0.99) |
| ark:quest:0 | 0.226 (9.25) | 0.241 (6.78) | 0.225 [0.218, 0.231] (4.38) | 0.174 [0.168, 0.182] (2.50) | 0.049 (0.94) |
| ark:cuboid_mean:1 | -0.365 (10.47) | -0.435 (8.20) | -0.345 [-0.368, -0.322] (5.32) | -0.183 [-0.203, -0.162] (2.87) | -0.193 (1.01) |
| ark:cuboid_mean:0 | -0.331 (10.40) | -0.403 (8.13) | -0.302 [-0.326, -0.278] (5.25) | -0.111 [-0.131, -0.089] (2.80) | -0.013 (0.96) |
| ark:digest(lodo):1 | 0.188 (9.32) | 0.202 (6.86) | 0.176 [0.171, 0.180] (4.46) | 0.099 [0.092, 0.105] (2.58) | -0.128 (0.99) |
| belady | 1.000 (7.64) | 1.000 (5.19) | 1.000 (3.10) | 1.000 (1.65) | 1.000 (0.69) |

### stream `plain`, scope `b` — G (miss rate %)

documents with a feasible equal-slots budget: B=128: 7, B=192: 14, B=256: 14, B=384: 14, B=512: 14

| policy | B=128 | B=192 | B=256 | B=384 | B=512 |
|---|---|---|---|---|---|
| lru | 0.000 (8.54) | 0.000 (2.79) | 0.000 (1.37) | 0.000 (0.76) | 0.000 (0.64) |
| score | 0.232 [0.220, 0.244] (8.10) | 0.169 (2.61) | 0.092 (1.32) | 0.032 (0.76) | 0.161 (0.63) |
| ema_z(lodo) | 0.265 [0.254, 0.276] (8.04) | 0.248 (2.53) | 0.227 (1.26) | 0.231 (0.73) | 0.341 (0.62) |
| ema_allrank(lodo) | 0.260 [0.250, 0.270] (8.05) | 0.235 (2.54) | 0.207 (1.27) | 0.212 (0.74) | 0.340 (0.62) |
| powerlaw(lodo) | 0.288 [0.280, 0.296] (8.00) | 0.283 (2.49) | 0.240 (1.26) | 0.220 (0.73) | 0.322 (0.62) |
| arkvale_simple | 0.145 [0.104, 0.183] (8.27) | 0.154 (2.62) | 0.089 (1.33) | 0.035 (0.76) | 0.163 (0.63) |
| ark:quest:1 | 0.232 [0.220, 0.244] (8.10) | 0.169 (2.61) | 0.092 (1.32) | 0.032 (0.76) | 0.161 (0.63) |
| ark:quest:0 | 0.232 [0.220, 0.244] (8.10) | 0.169 (2.61) | 0.092 (1.32) | 0.032 (0.76) | 0.161 (0.63) |
| ark:cuboid_mean:1 | -0.347 [-0.376, -0.308] (9.19) | -0.141 (2.93) | -0.039 (1.39) | 0.004 (0.76) | 0.173 (0.63) |
| ark:cuboid_mean:0 | -0.347 [-0.376, -0.308] (9.19) | -0.141 (2.93) | -0.039 (1.39) | 0.004 (0.76) | 0.173 (0.63) |
| ark:digest(lodo):1 | 0.232 [0.220, 0.244] (8.10) | 0.169 (2.61) | 0.092 (1.32) | 0.032 (0.76) | 0.173 (0.63) |
| belady | 1.000 (6.67) | 1.000 (1.74) | 1.000 (0.90) | 1.000 (0.63) | 1.000 (0.60) |

### stream `page`, scope `b` — G (miss rate %)

documents with a feasible equal-slots budget: B=128: 6, B=192: 14, B=256: 14, B=384: 14, B=512: 14

| policy | B=128 | B=192 | B=256 | B=384 | B=512 |
|---|---|---|---|---|---|
| lru | 0.000 (7.83) | 0.000 (2.78) | 0.000 (1.36) | 0.000 (0.76) | 0.000 (0.64) |
| score | 0.236 [0.224, 0.248] (7.34) | 0.170 (2.60) | 0.093 (1.32) | 0.031 (0.76) | 0.161 (0.63) |
| ema_z(lodo) | 0.267 [0.257, 0.278] (7.28) | 0.249 (2.52) | 0.227 (1.26) | 0.231 (0.73) | 0.341 (0.62) |
| ema_allrank(lodo) | 0.262 [0.251, 0.273] (7.29) | 0.236 (2.53) | 0.207 (1.27) | 0.212 (0.73) | 0.340 (0.62) |
| powerlaw(lodo) | 0.289 [0.281, 0.298] (7.23) | 0.284 (2.48) | 0.240 (1.25) | 0.220 (0.73) | 0.322 (0.62) |
| arkvale_simple | 0.161 [0.129, 0.195] (7.50) | 0.154 (2.62) | 0.089 (1.32) | 0.035 (0.76) | 0.161 (0.63) |
| ark:quest:1 | 0.193 [0.182, 0.205] (7.43) | 0.092 (2.68) | -0.045 (1.39) | -0.202 (0.79) | -0.211 (0.64) |
| ark:quest:0 | 0.236 [0.224, 0.248] (7.34) | 0.170 (2.60) | 0.093 (1.32) | 0.031 (0.76) | 0.161 (0.63) |
| ark:cuboid_mean:1 | -0.399 [-0.421, -0.376] (8.65) | -0.218 (3.01) | -0.177 (1.45) | -0.230 (0.79) | -0.198 (0.64) |
| ark:cuboid_mean:0 | -0.359 [-0.382, -0.332] (8.57) | -0.142 (2.93) | -0.039 (1.38) | 0.004 (0.76) | 0.173 (0.63) |
| ark:digest(lodo):1 | 0.193 [0.182, 0.205] (7.43) | 0.092 (2.68) | -0.045 (1.39) | -0.202 (0.79) | -0.198 (0.64) |
| belady | 1.000 (5.77) | 1.000 (1.73) | 1.000 (0.90) | 1.000 (0.63) | 1.000 (0.60) |

### stream `token`, scope `b` — G (miss rate %)

documents with a feasible equal-slots budget: B=128: 6, B=192: 14, B=256: 14, B=384: 14, B=512: 14

| policy | B=128 | B=192 | B=256 | B=384 | B=512 |
|---|---|---|---|---|---|
| lru | 0.000 (7.95) | 0.000 (2.76) | 0.000 (1.35) | 0.000 (0.75) | 0.000 (0.63) |
| score | 0.235 [0.222, 0.248] (7.48) | 0.174 (2.58) | 0.096 (1.30) | 0.035 (0.75) | 0.160 (0.62) |
| ema_z(lodo) | 0.266 [0.255, 0.277] (7.42) | 0.250 (2.50) | 0.228 (1.24) | 0.232 (0.72) | 0.341 (0.61) |
| ema_allrank(lodo) | 0.261 [0.250, 0.272] (7.42) | 0.238 (2.52) | 0.208 (1.25) | 0.213 (0.72) | 0.341 (0.61) |
| powerlaw(lodo) | 0.288 [0.280, 0.297] (7.37) | 0.284 (2.47) | 0.241 (1.24) | 0.221 (0.72) | 0.323 (0.61) |
| arkvale_simple | 0.235 [0.222, 0.248] (7.48) | 0.174 (2.58) | 0.096 (1.30) | 0.035 (0.75) | 0.160 (0.62) |
| ark:quest:1 | 0.195 [0.183, 0.207] (7.56) | 0.100 (2.66) | -0.038 (1.37) | -0.195 (0.78) | -0.201 (0.63) |
| ark:quest:0 | 0.235 [0.222, 0.248] (7.48) | 0.174 (2.58) | 0.096 (1.30) | 0.035 (0.75) | 0.160 (0.62) |
| ark:cuboid_mean:1 | -0.387 [-0.408, -0.364] (8.72) | -0.215 (2.99) | -0.171 (1.43) | -0.229 (0.78) | -0.195 (0.63) |
| ark:cuboid_mean:0 | -0.350 [-0.373, -0.323] (8.65) | -0.144 (2.91) | -0.037 (1.36) | 0.003 (0.75) | 0.171 (0.62) |
| ark:digest(lodo):1 | 0.195 [0.183, 0.207] (7.56) | 0.100 (2.66) | -0.038 (1.37) | -0.195 (0.78) | -0.195 (0.63) |
| belady | 1.000 (5.95) | 1.000 (1.73) | 1.000 (0.88) | 1.000 (0.62) | 1.000 (0.59) |

### Paired contrasts (dG = G(a) - G(b), 95 % document bootstrap; docs a better / b better)

| stream | scope | B | a | b | dG [CI] | docs |
|---|---|---|---|---|---|---|
| plain | all | 72 | ark:cuboid_mean:1 | ema_z(lodo) | -0.624 [-0.640, -0.608] | 0/14 of 14 |
| plain | all | 72 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.619 [-0.635, -0.603] | 0/14 of 14 |
| plain | all | 72 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.648 [-0.664, -0.633] | 0/14 of 14 |
| plain | all | 72 | ark:cuboid_mean:1 | score | -0.592 [-0.608, -0.576] | 0/14 of 14 |
| plain | all | 72 | ema_z(lodo) | score | +0.032 [+0.030, +0.034] | 14/0 of 14 |
| plain | all | 72 | ema_z(lodo) | lru | +0.258 [+0.255, +0.261] | 14/0 of 14 |
| plain | all | 72 | ark:cuboid_mean:1 | arkvale_simple | -0.486 [-0.500, -0.472] | 0/14 of 14 |
| plain | all | 72 | ark:digest(lodo):1 | ema_z(lodo) | -0.032 [-0.034, -0.030] | 0/14 of 14 |
| plain | all | 72 | ark:quest:1 | ema_z(lodo) | -0.032 [-0.034, -0.030] | 0/14 of 14 |
| plain | all | 72 | ark:cuboid_mean:1 | ark:quest:1 | -0.592 [-0.608, -0.576] | 0/14 of 14 |
| plain | all | 72 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | +0.000 [+0.000, +0.000] | 0/0 of 14 |
| plain | all | 80 | ark:cuboid_mean:1 | ema_z(lodo) | -0.694 [-0.716, -0.673] | 0/14 of 14 |
| plain | all | 80 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.687 [-0.708, -0.666] | 0/14 of 14 |
| plain | all | 80 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.712 [-0.733, -0.690] | 0/14 of 14 |
| plain | all | 80 | ark:cuboid_mean:1 | score | -0.657 [-0.678, -0.636] | 0/14 of 14 |
| plain | all | 80 | ema_z(lodo) | score | +0.038 [+0.035, +0.040] | 14/0 of 14 |
| plain | all | 80 | ema_z(lodo) | lru | +0.270 [+0.267, +0.274] | 14/0 of 14 |
| plain | all | 80 | ark:cuboid_mean:1 | arkvale_simple | -0.599 [-0.619, -0.579] | 0/14 of 14 |
| plain | all | 80 | ark:digest(lodo):1 | ema_z(lodo) | -0.038 [-0.040, -0.035] | 0/14 of 14 |
| plain | all | 80 | ark:quest:1 | ema_z(lodo) | -0.038 [-0.040, -0.035] | 0/14 of 14 |
| plain | all | 80 | ark:cuboid_mean:1 | ark:quest:1 | -0.657 [-0.678, -0.636] | 0/14 of 14 |
| plain | all | 80 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | +0.000 [+0.000, +0.000] | 0/0 of 14 |
| plain | all | 96 | ark:cuboid_mean:1 | ema_z(lodo) | -0.569 [-0.594, -0.545] | 0/14 of 14 |
| plain | all | 96 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.560 [-0.584, -0.535] | 0/14 of 14 |
| plain | all | 96 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.608 [-0.631, -0.585] | 0/14 of 14 |
| plain | all | 96 | ark:cuboid_mean:1 | score | -0.518 [-0.543, -0.494] | 0/14 of 14 |
| plain | all | 96 | ema_z(lodo) | score | +0.051 [+0.047, +0.054] | 14/0 of 14 |
| plain | all | 96 | ema_z(lodo) | lru | +0.272 [+0.268, +0.276] | 14/0 of 14 |
| plain | all | 96 | ark:cuboid_mean:1 | arkvale_simple | -0.490 [-0.515, -0.466] | 0/14 of 14 |
| plain | all | 96 | ark:digest(lodo):1 | ema_z(lodo) | -0.051 [-0.054, -0.047] | 0/14 of 14 |
| plain | all | 96 | ark:quest:1 | ema_z(lodo) | -0.051 [-0.054, -0.047] | 0/14 of 14 |
| plain | all | 96 | ark:cuboid_mean:1 | ark:quest:1 | -0.518 [-0.543, -0.494] | 0/14 of 14 |
| plain | all | 96 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | +0.000 [+0.000, +0.000] | 0/0 of 14 |
| plain | all | 128 | ark:cuboid_mean:1 | ema_z(lodo) | -0.361 [-0.382, -0.336] | 0/14 of 14 |
| plain | all | 128 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.346 [-0.367, -0.320] | 0/14 of 14 |
| plain | all | 128 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.390 [-0.409, -0.368] | 0/14 of 14 |
| plain | all | 128 | ark:cuboid_mean:1 | score | -0.277 [-0.297, -0.254] | 0/14 of 14 |
| plain | all | 128 | ema_z(lodo) | score | +0.084 [+0.079, +0.090] | 14/0 of 14 |
| plain | all | 128 | ema_z(lodo) | lru | +0.256 [+0.250, +0.262] | 14/0 of 14 |
| plain | all | 128 | ark:cuboid_mean:1 | arkvale_simple | -0.266 [-0.286, -0.244] | 0/14 of 14 |
| plain | all | 128 | ark:digest(lodo):1 | ema_z(lodo) | -0.084 [-0.090, -0.079] | 0/14 of 14 |
| plain | all | 128 | ark:quest:1 | ema_z(lodo) | -0.084 [-0.090, -0.079] | 0/14 of 14 |
| plain | all | 128 | ark:cuboid_mean:1 | ark:quest:1 | -0.277 [-0.297, -0.254] | 0/14 of 14 |
| plain | all | 128 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | +0.000 [+0.000, +0.000] | 0/0 of 14 |
| plain | all | 256 | ark:cuboid_mean:1 | ema_z(lodo) | -0.244 [-0.263, -0.223] | 0/14 of 14 |
| plain | all | 256 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.219 [-0.236, -0.200] | 0/14 of 14 |
| plain | all | 256 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.238 [-0.256, -0.220] | 0/14 of 14 |
| plain | all | 256 | ark:cuboid_mean:1 | score | -0.061 [-0.079, -0.038] | 1/13 of 14 |
| plain | all | 256 | ema_z(lodo) | score | +0.183 [+0.164, +0.206] | 14/0 of 14 |
| plain | all | 256 | ema_z(lodo) | lru | +0.232 [+0.219, +0.246] | 14/0 of 14 |
| plain | all | 256 | ark:cuboid_mean:1 | arkvale_simple | -0.062 [-0.078, -0.041] | 1/13 of 14 |
| plain | all | 256 | ark:digest(lodo):1 | ema_z(lodo) | -0.183 [-0.206, -0.164] | 0/14 of 14 |
| plain | all | 256 | ark:quest:1 | ema_z(lodo) | -0.183 [-0.206, -0.164] | 0/14 of 14 |
| plain | all | 256 | ark:cuboid_mean:1 | ark:quest:1 | -0.061 [-0.079, -0.038] | 1/13 of 14 |
| plain | all | 256 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | +0.000 [+0.000, +0.000] | 0/0 of 14 |
| plain | a | 72 | ark:cuboid_mean:1 | ema_z(lodo) | -0.609 [-0.625, -0.593] | 0/14 of 14 |
| plain | a | 72 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.604 [-0.620, -0.588] | 0/14 of 14 |
| plain | a | 72 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.635 [-0.651, -0.620] | 0/14 of 14 |
| plain | a | 72 | ark:cuboid_mean:1 | score | -0.580 [-0.595, -0.564] | 0/14 of 14 |
| plain | a | 72 | ema_z(lodo) | score | +0.029 [+0.027, +0.031] | 14/0 of 14 |
| plain | a | 72 | ema_z(lodo) | lru | +0.259 [+0.255, +0.262] | 14/0 of 14 |
| plain | a | 72 | ark:cuboid_mean:1 | arkvale_simple | -0.472 [-0.487, -0.459] | 0/14 of 14 |
| plain | a | 72 | ark:digest(lodo):1 | ema_z(lodo) | -0.029 [-0.031, -0.027] | 0/14 of 14 |
| plain | a | 72 | ark:quest:1 | ema_z(lodo) | -0.029 [-0.031, -0.027] | 0/14 of 14 |
| plain | a | 72 | ark:cuboid_mean:1 | ark:quest:1 | -0.580 [-0.595, -0.564] | 0/14 of 14 |
| plain | a | 72 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | +0.000 [+0.000, +0.000] | 0/0 of 14 |
| plain | a | 80 | ark:cuboid_mean:1 | ema_z(lodo) | -0.674 [-0.695, -0.653] | 0/14 of 14 |
| plain | a | 80 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.667 [-0.688, -0.647] | 0/14 of 14 |
| plain | a | 80 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.692 [-0.713, -0.671] | 0/14 of 14 |
| plain | a | 80 | ark:cuboid_mean:1 | score | -0.640 [-0.661, -0.619] | 0/14 of 14 |
| plain | a | 80 | ema_z(lodo) | score | +0.034 [+0.032, +0.036] | 14/0 of 14 |
| plain | a | 80 | ema_z(lodo) | lru | +0.273 [+0.269, +0.277] | 14/0 of 14 |
| plain | a | 80 | ark:cuboid_mean:1 | arkvale_simple | -0.582 [-0.602, -0.561] | 0/14 of 14 |
| plain | a | 80 | ark:digest(lodo):1 | ema_z(lodo) | -0.034 [-0.036, -0.032] | 0/14 of 14 |
| plain | a | 80 | ark:quest:1 | ema_z(lodo) | -0.034 [-0.036, -0.032] | 0/14 of 14 |
| plain | a | 80 | ark:cuboid_mean:1 | ark:quest:1 | -0.640 [-0.661, -0.619] | 0/14 of 14 |
| plain | a | 80 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | +0.000 [+0.000, +0.000] | 0/0 of 14 |
| plain | a | 96 | ark:cuboid_mean:1 | ema_z(lodo) | -0.564 [-0.589, -0.540] | 0/14 of 14 |
| plain | a | 96 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.555 [-0.580, -0.530] | 0/14 of 14 |
| plain | a | 96 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.604 [-0.627, -0.581] | 0/14 of 14 |
| plain | a | 96 | ark:cuboid_mean:1 | score | -0.514 [-0.539, -0.489] | 0/14 of 14 |
| plain | a | 96 | ema_z(lodo) | score | +0.050 [+0.047, +0.054] | 14/0 of 14 |
| plain | a | 96 | ema_z(lodo) | lru | +0.271 [+0.266, +0.275] | 14/0 of 14 |
| plain | a | 96 | ark:cuboid_mean:1 | arkvale_simple | -0.486 [-0.510, -0.460] | 0/14 of 14 |
| plain | a | 96 | ark:digest(lodo):1 | ema_z(lodo) | -0.050 [-0.054, -0.047] | 0/14 of 14 |
| plain | a | 96 | ark:quest:1 | ema_z(lodo) | -0.050 [-0.054, -0.047] | 0/14 of 14 |
| plain | a | 96 | ark:cuboid_mean:1 | ark:quest:1 | -0.514 [-0.539, -0.489] | 0/14 of 14 |
| plain | a | 96 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | +0.000 [+0.000, +0.000] | 0/0 of 14 |
| plain | a | 128 | ark:cuboid_mean:1 | ema_z(lodo) | -0.362 [-0.383, -0.337] | 0/14 of 14 |
| plain | a | 128 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.347 [-0.368, -0.321] | 0/14 of 14 |
| plain | a | 128 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.391 [-0.410, -0.369] | 0/14 of 14 |
| plain | a | 128 | ark:cuboid_mean:1 | score | -0.277 [-0.298, -0.253] | 0/14 of 14 |
| plain | a | 128 | ema_z(lodo) | score | +0.085 [+0.080, +0.091] | 14/0 of 14 |
| plain | a | 128 | ema_z(lodo) | lru | +0.254 [+0.248, +0.260] | 14/0 of 14 |
| plain | a | 128 | ark:cuboid_mean:1 | arkvale_simple | -0.266 [-0.286, -0.244] | 0/14 of 14 |
| plain | a | 128 | ark:digest(lodo):1 | ema_z(lodo) | -0.085 [-0.091, -0.080] | 0/14 of 14 |
| plain | a | 128 | ark:quest:1 | ema_z(lodo) | -0.085 [-0.091, -0.080] | 0/14 of 14 |
| plain | a | 128 | ark:cuboid_mean:1 | ark:quest:1 | -0.277 [-0.298, -0.253] | 0/14 of 14 |
| plain | a | 128 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | +0.000 [+0.000, +0.000] | 0/0 of 14 |
| plain | a | 256 | ark:cuboid_mean:1 | ema_z(lodo) | -0.244 [-0.264, -0.224] | 0/14 of 14 |
| plain | a | 256 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.219 [-0.236, -0.201] | 0/14 of 14 |
| plain | a | 256 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.238 [-0.256, -0.220] | 0/14 of 14 |
| plain | a | 256 | ark:cuboid_mean:1 | score | -0.061 [-0.079, -0.038] | 1/13 of 14 |
| plain | a | 256 | ema_z(lodo) | score | +0.184 [+0.165, +0.207] | 14/0 of 14 |
| plain | a | 256 | ema_z(lodo) | lru | +0.229 [+0.216, +0.243] | 14/0 of 14 |
| plain | a | 256 | ark:cuboid_mean:1 | arkvale_simple | -0.061 [-0.078, -0.041] | 1/13 of 14 |
| plain | a | 256 | ark:digest(lodo):1 | ema_z(lodo) | -0.184 [-0.207, -0.165] | 0/14 of 14 |
| plain | a | 256 | ark:quest:1 | ema_z(lodo) | -0.184 [-0.207, -0.165] | 0/14 of 14 |
| plain | a | 256 | ark:cuboid_mean:1 | ark:quest:1 | -0.061 [-0.079, -0.038] | 1/13 of 14 |
| plain | a | 256 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | +0.000 [+0.000, +0.000] | 0/0 of 14 |
| plain | b | 128 | ark:cuboid_mean:1 | ema_z(lodo) | -0.613 [-0.648, -0.567] | 0/7 of 7 |
| plain | b | 128 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.608 [-0.643, -0.563] | 0/7 of 7 |
| plain | b | 128 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.635 [-0.669, -0.592] | 0/7 of 7 |
| plain | b | 128 | ark:cuboid_mean:1 | score | -0.580 [-0.618, -0.532] | 0/7 of 7 |
| plain | b | 128 | ema_z(lodo) | score | +0.033 [+0.029, +0.037] | 7/0 of 7 |
| plain | b | 128 | ema_z(lodo) | lru | +0.265 [+0.254, +0.276] | 7/0 of 7 |
| plain | b | 128 | ark:cuboid_mean:1 | arkvale_simple | -0.493 [-0.547, -0.424] | 0/7 of 7 |
| plain | b | 128 | ark:digest(lodo):1 | ema_z(lodo) | -0.033 [-0.037, -0.029] | 0/7 of 7 |
| plain | b | 128 | ark:quest:1 | ema_z(lodo) | -0.033 [-0.037, -0.029] | 0/7 of 7 |
| plain | b | 128 | ark:cuboid_mean:1 | ark:quest:1 | -0.580 [-0.618, -0.532] | 0/7 of 7 |
| plain | b | 128 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | +0.000 [+0.000, +0.000] | 0/0 of 7 |
| plain | b | 192 | ark:cuboid_mean:1 | ema_z(lodo) | -0.390 [-0.447, -0.332] | 0/14 of 14 |
| plain | b | 192 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.376 [-0.432, -0.320] | 0/14 of 14 |
| plain | b | 192 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.424 [-0.486, -0.364] | 0/14 of 14 |
| plain | b | 192 | ark:cuboid_mean:1 | score | -0.310 [-0.377, -0.242] | 0/14 of 14 |
| plain | b | 192 | ema_z(lodo) | score | +0.080 [+0.069, +0.092] | 14/0 of 14 |
| plain | b | 192 | ema_z(lodo) | lru | +0.248 [+0.238, +0.256] | 14/0 of 14 |
| plain | b | 192 | ark:cuboid_mean:1 | arkvale_simple | -0.296 [-0.358, -0.232] | 0/14 of 14 |
| plain | b | 192 | ark:digest(lodo):1 | ema_z(lodo) | -0.080 [-0.092, -0.069] | 0/14 of 14 |
| plain | b | 192 | ark:quest:1 | ema_z(lodo) | -0.080 [-0.092, -0.069] | 0/14 of 14 |
| plain | b | 192 | ark:cuboid_mean:1 | ark:quest:1 | -0.310 [-0.377, -0.242] | 0/14 of 14 |
| plain | b | 192 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | +0.000 [+0.000, +0.000] | 0/0 of 14 |
| plain | b | 256 | ark:cuboid_mean:1 | ema_z(lodo) | -0.266 [-0.285, -0.245] | 0/14 of 14 |
| plain | b | 256 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.246 [-0.265, -0.224] | 0/14 of 14 |
| plain | b | 256 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.279 [-0.301, -0.254] | 0/14 of 14 |
| plain | b | 256 | ark:cuboid_mean:1 | score | -0.131 [-0.157, -0.100] | 1/13 of 14 |
| plain | b | 256 | ema_z(lodo) | score | +0.136 [+0.120, +0.155] | 14/0 of 14 |
| plain | b | 256 | ema_z(lodo) | lru | +0.227 [+0.216, +0.240] | 14/0 of 14 |
| plain | b | 256 | ark:cuboid_mean:1 | arkvale_simple | -0.128 [-0.153, -0.099] | 1/13 of 14 |
| plain | b | 256 | ark:digest(lodo):1 | ema_z(lodo) | -0.136 [-0.155, -0.120] | 0/14 of 14 |
| plain | b | 256 | ark:quest:1 | ema_z(lodo) | -0.136 [-0.155, -0.120] | 0/14 of 14 |
| plain | b | 256 | ark:cuboid_mean:1 | ark:quest:1 | -0.131 [-0.157, -0.100] | 1/13 of 14 |
| plain | b | 256 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | +0.000 [+0.000, +0.000] | 0/0 of 14 |
| plain | b | 384 | ark:cuboid_mean:1 | ema_z(lodo) | -0.227 [-0.246, -0.207] | 0/14 of 14 |
| plain | b | 384 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.208 [-0.225, -0.190] | 0/14 of 14 |
| plain | b | 384 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.217 [-0.237, -0.196] | 0/14 of 14 |
| plain | b | 384 | ark:cuboid_mean:1 | score | -0.028 [-0.045, -0.010] | 5/9 of 14 |
| plain | b | 384 | ema_z(lodo) | score | +0.199 [+0.182, +0.216] | 14/0 of 14 |
| plain | b | 384 | ema_z(lodo) | lru | +0.231 [+0.211, +0.252] | 14/0 of 14 |
| plain | b | 384 | ark:cuboid_mean:1 | arkvale_simple | -0.031 [-0.048, -0.014] | 3/11 of 14 |
| plain | b | 384 | ark:digest(lodo):1 | ema_z(lodo) | -0.199 [-0.216, -0.182] | 0/14 of 14 |
| plain | b | 384 | ark:quest:1 | ema_z(lodo) | -0.199 [-0.216, -0.182] | 0/14 of 14 |
| plain | b | 384 | ark:cuboid_mean:1 | ark:quest:1 | -0.028 [-0.045, -0.010] | 5/9 of 14 |
| plain | b | 384 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | +0.000 [+0.000, +0.000] | 0/0 of 14 |
| plain | b | 512 | ark:cuboid_mean:1 | ema_z(lodo) | -0.168 [-0.205, -0.121] | 0/14 of 14 |
| plain | b | 512 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.167 [-0.198, -0.129] | 0/14 of 14 |
| plain | b | 512 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.150 [-0.189, -0.101] | 0/14 of 14 |
| plain | b | 512 | ark:cuboid_mean:1 | score | +0.012 [-0.011, +0.033] | 8/5 of 14 |
| plain | b | 512 | ema_z(lodo) | score | +0.180 [+0.135, +0.218] | 14/0 of 14 |
| plain | b | 512 | ema_z(lodo) | lru | +0.341 [+0.296, +0.399] | 14/0 of 14 |
| plain | b | 512 | ark:cuboid_mean:1 | arkvale_simple | +0.010 [-0.012, +0.030] | 8/5 of 14 |
| plain | b | 512 | ark:digest(lodo):1 | ema_z(lodo) | -0.168 [-0.205, -0.121] | 0/14 of 14 |
| plain | b | 512 | ark:quest:1 | ema_z(lodo) | -0.180 [-0.218, -0.135] | 0/14 of 14 |
| plain | b | 512 | ark:cuboid_mean:1 | ark:quest:1 | +0.012 [-0.011, +0.033] | 8/5 of 14 |
| plain | b | 512 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | +0.000 [+0.000, +0.000] | 0/0 of 14 |
| page | all | 72 | ark:cuboid_mean:1 | ema_z(lodo) | -0.659 [-0.674, -0.643] | 0/14 of 14 |
| page | all | 72 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.654 [-0.670, -0.638] | 0/14 of 14 |
| page | all | 72 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.683 [-0.699, -0.668] | 0/14 of 14 |
| page | all | 72 | ark:cuboid_mean:1 | score | -0.627 [-0.643, -0.612] | 0/14 of 14 |
| page | all | 72 | ema_z(lodo) | score | +0.031 [+0.029, +0.033] | 14/0 of 14 |
| page | all | 72 | ema_z(lodo) | lru | +0.259 [+0.255, +0.262] | 14/0 of 14 |
| page | all | 72 | ark:cuboid_mean:1 | arkvale_simple | -0.533 [-0.547, -0.519] | 0/14 of 14 |
| page | all | 72 | ark:digest(lodo):1 | ema_z(lodo) | -0.072 [-0.074, -0.070] | 0/14 of 14 |
| page | all | 72 | ark:quest:1 | ema_z(lodo) | -0.072 [-0.074, -0.070] | 0/14 of 14 |
| page | all | 72 | ark:cuboid_mean:1 | ark:quest:1 | -0.587 [-0.602, -0.572] | 0/14 of 14 |
| page | all | 72 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.038 [-0.040, -0.036] | 0/14 of 14 |
| page | all | 80 | ark:cuboid_mean:1 | ema_z(lodo) | -0.736 [-0.757, -0.714] | 0/14 of 14 |
| page | all | 80 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.729 [-0.750, -0.708] | 0/14 of 14 |
| page | all | 80 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.753 [-0.773, -0.731] | 0/14 of 14 |
| page | all | 80 | ark:cuboid_mean:1 | score | -0.698 [-0.719, -0.677] | 0/14 of 14 |
| page | all | 80 | ema_z(lodo) | score | +0.037 [+0.035, +0.040] | 14/0 of 14 |
| page | all | 80 | ema_z(lodo) | lru | +0.271 [+0.267, +0.275] | 14/0 of 14 |
| page | all | 80 | ark:cuboid_mean:1 | arkvale_simple | -0.644 [-0.664, -0.623] | 0/14 of 14 |
| page | all | 80 | ark:digest(lodo):1 | ema_z(lodo) | -0.080 [-0.083, -0.078] | 0/14 of 14 |
| page | all | 80 | ark:quest:1 | ema_z(lodo) | -0.080 [-0.083, -0.078] | 0/14 of 14 |
| page | all | 80 | ark:cuboid_mean:1 | ark:quest:1 | -0.655 [-0.676, -0.635] | 0/14 of 14 |
| page | all | 80 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.039 [-0.041, -0.037] | 0/14 of 14 |
| page | all | 96 | ark:cuboid_mean:1 | ema_z(lodo) | -0.623 [-0.648, -0.599] | 0/14 of 14 |
| page | all | 96 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.614 [-0.639, -0.590] | 0/14 of 14 |
| page | all | 96 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.662 [-0.686, -0.639] | 0/14 of 14 |
| page | all | 96 | ark:cuboid_mean:1 | score | -0.573 [-0.598, -0.548] | 0/14 of 14 |
| page | all | 96 | ema_z(lodo) | score | +0.050 [+0.047, +0.053] | 14/0 of 14 |
| page | all | 96 | ema_z(lodo) | lru | +0.272 [+0.268, +0.276] | 14/0 of 14 |
| page | all | 96 | ark:cuboid_mean:1 | arkvale_simple | -0.545 [-0.569, -0.520] | 0/14 of 14 |
| page | all | 96 | ark:digest(lodo):1 | ema_z(lodo) | -0.105 [-0.108, -0.101] | 0/14 of 14 |
| page | all | 96 | ark:quest:1 | ema_z(lodo) | -0.105 [-0.108, -0.101] | 0/14 of 14 |
| page | all | 96 | ark:cuboid_mean:1 | ark:quest:1 | -0.519 [-0.543, -0.494] | 0/14 of 14 |
| page | all | 96 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.051 [-0.054, -0.049] | 0/14 of 14 |
| page | all | 128 | ark:cuboid_mean:1 | ema_z(lodo) | -0.443 [-0.464, -0.419] | 0/14 of 14 |
| page | all | 128 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.428 [-0.450, -0.403] | 0/14 of 14 |
| page | all | 128 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.472 [-0.491, -0.450] | 0/14 of 14 |
| page | all | 128 | ark:cuboid_mean:1 | score | -0.360 [-0.381, -0.336] | 0/14 of 14 |
| page | all | 128 | ema_z(lodo) | score | +0.083 [+0.078, +0.089] | 14/0 of 14 |
| page | all | 128 | ema_z(lodo) | lru | +0.257 [+0.251, +0.262] | 14/0 of 14 |
| page | all | 128 | ark:cuboid_mean:1 | arkvale_simple | -0.348 [-0.368, -0.326] | 0/14 of 14 |
| page | all | 128 | ark:digest(lodo):1 | ema_z(lodo) | -0.166 [-0.173, -0.160] | 0/14 of 14 |
| page | all | 128 | ark:quest:1 | ema_z(lodo) | -0.166 [-0.173, -0.160] | 0/14 of 14 |
| page | all | 128 | ark:cuboid_mean:1 | ark:quest:1 | -0.277 [-0.298, -0.255] | 0/14 of 14 |
| page | all | 128 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.082 [-0.086, -0.077] | 0/14 of 14 |
| page | all | 256 | ark:cuboid_mean:1 | ema_z(lodo) | -0.434 [-0.457, -0.409] | 0/14 of 14 |
| page | all | 256 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.409 [-0.431, -0.384] | 0/14 of 14 |
| page | all | 256 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.428 [-0.448, -0.406] | 0/14 of 14 |
| page | all | 256 | ark:cuboid_mean:1 | score | -0.251 [-0.269, -0.234] | 0/14 of 14 |
| page | all | 256 | ema_z(lodo) | score | +0.183 [+0.164, +0.206] | 14/0 of 14 |
| page | all | 256 | ema_z(lodo) | lru | +0.232 [+0.219, +0.246] | 14/0 of 14 |
| page | all | 256 | ark:cuboid_mean:1 | arkvale_simple | -0.251 [-0.268, -0.235] | 0/14 of 14 |
| page | all | 256 | ark:digest(lodo):1 | ema_z(lodo) | -0.370 [-0.403, -0.343] | 0/14 of 14 |
| page | all | 256 | ark:quest:1 | ema_z(lodo) | -0.370 [-0.403, -0.343] | 0/14 of 14 |
| page | all | 256 | ark:cuboid_mean:1 | ark:quest:1 | -0.063 [-0.082, -0.043] | 1/13 of 14 |
| page | all | 256 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.190 [-0.206, -0.175] | 0/14 of 14 |
| page | a | 72 | ark:cuboid_mean:1 | ema_z(lodo) | -0.645 [-0.661, -0.630] | 0/14 of 14 |
| page | a | 72 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.640 [-0.657, -0.625] | 0/14 of 14 |
| page | a | 72 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.671 [-0.687, -0.656] | 0/14 of 14 |
| page | a | 72 | ark:cuboid_mean:1 | score | -0.616 [-0.632, -0.601] | 0/14 of 14 |
| page | a | 72 | ema_z(lodo) | score | +0.029 [+0.027, +0.031] | 14/0 of 14 |
| page | a | 72 | ema_z(lodo) | lru | +0.258 [+0.255, +0.262] | 14/0 of 14 |
| page | a | 72 | ark:cuboid_mean:1 | arkvale_simple | -0.518 [-0.532, -0.504] | 0/14 of 14 |
| page | a | 72 | ark:digest(lodo):1 | ema_z(lodo) | -0.069 [-0.071, -0.067] | 0/14 of 14 |
| page | a | 72 | ark:quest:1 | ema_z(lodo) | -0.069 [-0.071, -0.067] | 0/14 of 14 |
| page | a | 72 | ark:cuboid_mean:1 | ark:quest:1 | -0.576 [-0.591, -0.562] | 0/14 of 14 |
| page | a | 72 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.037 [-0.039, -0.035] | 0/14 of 14 |
| page | a | 80 | ark:cuboid_mean:1 | ema_z(lodo) | -0.714 [-0.735, -0.693] | 0/14 of 14 |
| page | a | 80 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.707 [-0.729, -0.686] | 0/14 of 14 |
| page | a | 80 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.732 [-0.753, -0.711] | 0/14 of 14 |
| page | a | 80 | ark:cuboid_mean:1 | score | -0.680 [-0.701, -0.659] | 0/14 of 14 |
| page | a | 80 | ema_z(lodo) | score | +0.034 [+0.031, +0.036] | 14/0 of 14 |
| page | a | 80 | ema_z(lodo) | lru | +0.273 [+0.270, +0.277] | 14/0 of 14 |
| page | a | 80 | ark:cuboid_mean:1 | arkvale_simple | -0.624 [-0.644, -0.603] | 0/14 of 14 |
| page | a | 80 | ark:digest(lodo):1 | ema_z(lodo) | -0.075 [-0.077, -0.072] | 0/14 of 14 |
| page | a | 80 | ark:quest:1 | ema_z(lodo) | -0.075 [-0.077, -0.072] | 0/14 of 14 |
| page | a | 80 | ark:cuboid_mean:1 | ark:quest:1 | -0.639 [-0.659, -0.618] | 0/14 of 14 |
| page | a | 80 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.037 [-0.039, -0.035] | 0/14 of 14 |
| page | a | 96 | ark:cuboid_mean:1 | ema_z(lodo) | -0.614 [-0.639, -0.590] | 0/14 of 14 |
| page | a | 96 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.606 [-0.631, -0.581] | 0/14 of 14 |
| page | a | 96 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.654 [-0.678, -0.631] | 0/14 of 14 |
| page | a | 96 | ark:cuboid_mean:1 | score | -0.565 [-0.591, -0.540] | 0/14 of 14 |
| page | a | 96 | ema_z(lodo) | score | +0.049 [+0.045, +0.052] | 14/0 of 14 |
| page | a | 96 | ema_z(lodo) | lru | +0.270 [+0.266, +0.274] | 14/0 of 14 |
| page | a | 96 | ark:cuboid_mean:1 | arkvale_simple | -0.537 [-0.561, -0.512] | 0/14 of 14 |
| page | a | 96 | ark:digest(lodo):1 | ema_z(lodo) | -0.101 [-0.104, -0.097] | 0/14 of 14 |
| page | a | 96 | ark:quest:1 | ema_z(lodo) | -0.101 [-0.104, -0.097] | 0/14 of 14 |
| page | a | 96 | ark:cuboid_mean:1 | ark:quest:1 | -0.513 [-0.537, -0.488] | 0/14 of 14 |
| page | a | 96 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.049 [-0.052, -0.046] | 0/14 of 14 |
| page | a | 128 | ark:cuboid_mean:1 | ema_z(lodo) | -0.440 [-0.462, -0.416] | 0/14 of 14 |
| page | a | 128 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.426 [-0.448, -0.400] | 0/14 of 14 |
| page | a | 128 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.470 [-0.489, -0.448] | 0/14 of 14 |
| page | a | 128 | ark:cuboid_mean:1 | score | -0.356 [-0.378, -0.333] | 0/14 of 14 |
| page | a | 128 | ema_z(lodo) | score | +0.084 [+0.079, +0.090] | 14/0 of 14 |
| page | a | 128 | ema_z(lodo) | lru | +0.254 [+0.248, +0.260] | 14/0 of 14 |
| page | a | 128 | ark:cuboid_mean:1 | arkvale_simple | -0.344 [-0.365, -0.323] | 0/14 of 14 |
| page | a | 128 | ark:digest(lodo):1 | ema_z(lodo) | -0.163 [-0.170, -0.157] | 0/14 of 14 |
| page | a | 128 | ark:quest:1 | ema_z(lodo) | -0.163 [-0.170, -0.157] | 0/14 of 14 |
| page | a | 128 | ark:cuboid_mean:1 | ark:quest:1 | -0.277 [-0.298, -0.254] | 0/14 of 14 |
| page | a | 128 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.078 [-0.083, -0.073] | 0/14 of 14 |
| page | a | 256 | ark:cuboid_mean:1 | ema_z(lodo) | -0.427 [-0.450, -0.402] | 0/14 of 14 |
| page | a | 256 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.402 [-0.424, -0.377] | 0/14 of 14 |
| page | a | 256 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.421 [-0.441, -0.399] | 0/14 of 14 |
| page | a | 256 | ark:cuboid_mean:1 | score | -0.243 [-0.262, -0.226] | 0/14 of 14 |
| page | a | 256 | ema_z(lodo) | score | +0.183 [+0.165, +0.207] | 14/0 of 14 |
| page | a | 256 | ema_z(lodo) | lru | +0.230 [+0.216, +0.243] | 14/0 of 14 |
| page | a | 256 | ark:cuboid_mean:1 | arkvale_simple | -0.244 [-0.261, -0.227] | 0/14 of 14 |
| page | a | 256 | ark:digest(lodo):1 | ema_z(lodo) | -0.364 [-0.396, -0.337] | 0/14 of 14 |
| page | a | 256 | ark:quest:1 | ema_z(lodo) | -0.364 [-0.396, -0.337] | 0/14 of 14 |
| page | a | 256 | ark:cuboid_mean:1 | ark:quest:1 | -0.063 [-0.081, -0.042] | 1/13 of 14 |
| page | a | 256 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.183 [-0.198, -0.168] | 0/14 of 14 |
| page | b | 128 | ark:cuboid_mean:1 | ema_z(lodo) | -0.667 [-0.695, -0.641] | 0/6 of 6 |
| page | b | 128 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.661 [-0.689, -0.635] | 0/6 of 6 |
| page | b | 128 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.688 [-0.716, -0.663] | 0/6 of 6 |
| page | b | 128 | ark:cuboid_mean:1 | score | -0.635 [-0.666, -0.606] | 0/6 of 6 |
| page | b | 128 | ema_z(lodo) | score | +0.032 [+0.028, +0.036] | 6/0 of 6 |
| page | b | 128 | ema_z(lodo) | lru | +0.267 [+0.257, +0.278] | 6/0 of 6 |
| page | b | 128 | ark:cuboid_mean:1 | arkvale_simple | -0.560 [-0.602, -0.517] | 0/6 of 6 |
| page | b | 128 | ark:digest(lodo):1 | ema_z(lodo) | -0.074 [-0.080, -0.070] | 0/6 of 6 |
| page | b | 128 | ark:quest:1 | ema_z(lodo) | -0.074 [-0.080, -0.070] | 0/6 of 6 |
| page | b | 128 | ark:cuboid_mean:1 | ark:quest:1 | -0.592 [-0.623, -0.563] | 0/6 of 6 |
| page | b | 128 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.041 [-0.045, -0.037] | 0/6 of 6 |
| page | b | 192 | ark:cuboid_mean:1 | ema_z(lodo) | -0.467 [-0.515, -0.418] | 0/14 of 14 |
| page | b | 192 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.454 [-0.501, -0.407] | 0/14 of 14 |
| page | b | 192 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.501 [-0.553, -0.450] | 0/14 of 14 |
| page | b | 192 | ark:cuboid_mean:1 | score | -0.388 [-0.444, -0.329] | 0/14 of 14 |
| page | b | 192 | ema_z(lodo) | score | +0.079 [+0.068, +0.092] | 14/0 of 14 |
| page | b | 192 | ema_z(lodo) | lru | +0.249 [+0.239, +0.256] | 14/0 of 14 |
| page | b | 192 | ark:cuboid_mean:1 | arkvale_simple | -0.372 [-0.424, -0.319] | 0/14 of 14 |
| page | b | 192 | ark:digest(lodo):1 | ema_z(lodo) | -0.157 [-0.180, -0.135] | 0/14 of 14 |
| page | b | 192 | ark:quest:1 | ema_z(lodo) | -0.157 [-0.180, -0.135] | 0/14 of 14 |
| page | b | 192 | ark:cuboid_mean:1 | ark:quest:1 | -0.310 [-0.376, -0.242] | 0/14 of 14 |
| page | b | 192 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.076 [-0.089, -0.064] | 0/14 of 14 |
| page | b | 256 | ark:cuboid_mean:1 | ema_z(lodo) | -0.404 [-0.426, -0.383] | 0/14 of 14 |
| page | b | 256 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.384 [-0.405, -0.364] | 0/14 of 14 |
| page | b | 256 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.417 [-0.436, -0.396] | 0/14 of 14 |
| page | b | 256 | ark:cuboid_mean:1 | score | -0.270 [-0.287, -0.249] | 0/14 of 14 |
| page | b | 256 | ema_z(lodo) | score | +0.135 [+0.119, +0.153] | 14/0 of 14 |
| page | b | 256 | ema_z(lodo) | lru | +0.227 [+0.216, +0.240] | 14/0 of 14 |
| page | b | 256 | ark:cuboid_mean:1 | arkvale_simple | -0.266 [-0.283, -0.247] | 0/14 of 14 |
| page | b | 256 | ark:digest(lodo):1 | ema_z(lodo) | -0.272 [-0.309, -0.242] | 0/14 of 14 |
| page | b | 256 | ark:quest:1 | ema_z(lodo) | -0.272 [-0.309, -0.242] | 0/14 of 14 |
| page | b | 256 | ark:cuboid_mean:1 | ark:quest:1 | -0.132 [-0.158, -0.102] | 1/13 of 14 |
| page | b | 256 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.138 [-0.157, -0.120] | 0/14 of 14 |
| page | b | 384 | ark:cuboid_mean:1 | ema_z(lodo) | -0.461 [-0.491, -0.436] | 0/14 of 14 |
| page | b | 384 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.441 [-0.475, -0.415] | 0/14 of 14 |
| page | b | 384 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.450 [-0.484, -0.423] | 0/14 of 14 |
| page | b | 384 | ark:cuboid_mean:1 | score | -0.261 [-0.292, -0.233] | 0/14 of 14 |
| page | b | 384 | ema_z(lodo) | score | +0.199 [+0.182, +0.217] | 14/0 of 14 |
| page | b | 384 | ema_z(lodo) | lru | +0.231 [+0.211, +0.252] | 14/0 of 14 |
| page | b | 384 | ark:cuboid_mean:1 | arkvale_simple | -0.264 [-0.296, -0.237] | 0/14 of 14 |
| page | b | 384 | ark:digest(lodo):1 | ema_z(lodo) | -0.433 [-0.468, -0.405] | 0/14 of 14 |
| page | b | 384 | ark:quest:1 | ema_z(lodo) | -0.433 [-0.468, -0.405] | 0/14 of 14 |
| page | b | 384 | ark:cuboid_mean:1 | ark:quest:1 | -0.028 [-0.045, -0.009] | 4/10 of 14 |
| page | b | 384 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.234 [-0.267, -0.208] | 0/14 of 14 |
| page | b | 512 | ark:cuboid_mean:1 | ema_z(lodo) | -0.540 [-0.604, -0.496] | 0/14 of 14 |
| page | b | 512 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.539 [-0.605, -0.496] | 0/14 of 14 |
| page | b | 512 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.521 [-0.591, -0.472] | 0/14 of 14 |
| page | b | 512 | ark:cuboid_mean:1 | score | -0.359 [-0.433, -0.310] | 0/14 of 14 |
| page | b | 512 | ema_z(lodo) | score | +0.181 [+0.136, +0.219] | 14/0 of 14 |
| page | b | 512 | ema_z(lodo) | lru | +0.341 [+0.296, +0.399] | 14/0 of 14 |
| page | b | 512 | ark:cuboid_mean:1 | arkvale_simple | -0.359 [-0.433, -0.311] | 0/14 of 14 |
| page | b | 512 | ark:digest(lodo):1 | ema_z(lodo) | -0.540 [-0.604, -0.496] | 0/14 of 14 |
| page | b | 512 | ark:quest:1 | ema_z(lodo) | -0.552 [-0.615, -0.511] | 0/14 of 14 |
| page | b | 512 | ark:cuboid_mean:1 | ark:quest:1 | +0.012 [-0.010, +0.034] | 9/5 of 14 |
| page | b | 512 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.372 [-0.439, -0.323] | 0/14 of 14 |
| token | all | 72 | ark:cuboid_mean:1 | ema_z(lodo) | -0.631 [-0.645, -0.618] | 0/14 of 14 |
| token | all | 72 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.626 [-0.640, -0.613] | 0/14 of 14 |
| token | all | 72 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.657 [-0.670, -0.644] | 0/14 of 14 |
| token | all | 72 | ark:cuboid_mean:1 | score | -0.601 [-0.614, -0.587] | 0/14 of 14 |
| token | all | 72 | ema_z(lodo) | score | +0.030 [+0.028, +0.033] | 14/0 of 14 |
| token | all | 72 | ema_z(lodo) | lru | +0.255 [+0.253, +0.257] | 14/0 of 14 |
| token | all | 72 | ark:cuboid_mean:1 | arkvale_simple | -0.601 [-0.614, -0.587] | 0/14 of 14 |
| token | all | 72 | ark:digest(lodo):1 | ema_z(lodo) | -0.069 [-0.071, -0.067] | 0/14 of 14 |
| token | all | 72 | ark:quest:1 | ema_z(lodo) | -0.069 [-0.071, -0.067] | 0/14 of 14 |
| token | all | 72 | ark:cuboid_mean:1 | ark:quest:1 | -0.562 [-0.575, -0.548] | 0/14 of 14 |
| token | all | 72 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.035 [-0.037, -0.033] | 0/14 of 14 |
| token | all | 80 | ark:cuboid_mean:1 | ema_z(lodo) | -0.730 [-0.750, -0.709] | 0/14 of 14 |
| token | all | 80 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.724 [-0.744, -0.703] | 0/14 of 14 |
| token | all | 80 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.748 [-0.768, -0.728] | 0/14 of 14 |
| token | all | 80 | ark:cuboid_mean:1 | score | -0.694 [-0.714, -0.674] | 0/14 of 14 |
| token | all | 80 | ema_z(lodo) | score | +0.036 [+0.033, +0.038] | 14/0 of 14 |
| token | all | 80 | ema_z(lodo) | lru | +0.270 [+0.267, +0.274] | 14/0 of 14 |
| token | all | 80 | ark:cuboid_mean:1 | arkvale_simple | -0.694 [-0.714, -0.674] | 0/14 of 14 |
| token | all | 80 | ark:digest(lodo):1 | ema_z(lodo) | -0.075 [-0.078, -0.073] | 0/14 of 14 |
| token | all | 80 | ark:quest:1 | ema_z(lodo) | -0.075 [-0.078, -0.073] | 0/14 of 14 |
| token | all | 80 | ark:cuboid_mean:1 | ark:quest:1 | -0.654 [-0.674, -0.634] | 0/14 of 14 |
| token | all | 80 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.034 [-0.036, -0.031] | 0/14 of 14 |
| token | all | 96 | ark:cuboid_mean:1 | ema_z(lodo) | -0.627 [-0.651, -0.603] | 0/14 of 14 |
| token | all | 96 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.619 [-0.643, -0.595] | 0/14 of 14 |
| token | all | 96 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.666 [-0.689, -0.643] | 0/14 of 14 |
| token | all | 96 | ark:cuboid_mean:1 | score | -0.579 [-0.603, -0.554] | 0/14 of 14 |
| token | all | 96 | ema_z(lodo) | score | +0.049 [+0.045, +0.052] | 14/0 of 14 |
| token | all | 96 | ema_z(lodo) | lru | +0.273 [+0.269, +0.277] | 14/0 of 14 |
| token | all | 96 | ark:cuboid_mean:1 | arkvale_simple | -0.579 [-0.603, -0.554] | 0/14 of 14 |
| token | all | 96 | ark:digest(lodo):1 | ema_z(lodo) | -0.100 [-0.103, -0.097] | 0/14 of 14 |
| token | all | 96 | ark:quest:1 | ema_z(lodo) | -0.100 [-0.103, -0.097] | 0/14 of 14 |
| token | all | 96 | ark:cuboid_mean:1 | ark:quest:1 | -0.527 [-0.551, -0.503] | 0/14 of 14 |
| token | all | 96 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.045 [-0.048, -0.042] | 0/14 of 14 |
| token | all | 128 | ark:cuboid_mean:1 | ema_z(lodo) | -0.442 [-0.463, -0.419] | 0/14 of 14 |
| token | all | 128 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.428 [-0.449, -0.403] | 0/14 of 14 |
| token | all | 128 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.470 [-0.490, -0.449] | 0/14 of 14 |
| token | all | 128 | ark:cuboid_mean:1 | score | -0.361 [-0.382, -0.338] | 0/14 of 14 |
| token | all | 128 | ema_z(lodo) | score | +0.081 [+0.076, +0.086] | 14/0 of 14 |
| token | all | 128 | ema_z(lodo) | lru | +0.258 [+0.253, +0.263] | 14/0 of 14 |
| token | all | 128 | ark:cuboid_mean:1 | arkvale_simple | -0.361 [-0.382, -0.338] | 0/14 of 14 |
| token | all | 128 | ark:digest(lodo):1 | ema_z(lodo) | -0.160 [-0.166, -0.154] | 0/14 of 14 |
| token | all | 128 | ark:quest:1 | ema_z(lodo) | -0.160 [-0.166, -0.154] | 0/14 of 14 |
| token | all | 128 | ark:cuboid_mean:1 | ark:quest:1 | -0.282 [-0.302, -0.260] | 0/14 of 14 |
| token | all | 128 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.076 [-0.081, -0.071] | 0/14 of 14 |
| token | all | 256 | ark:cuboid_mean:1 | ema_z(lodo) | -0.429 [-0.452, -0.405] | 0/14 of 14 |
| token | all | 256 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.405 [-0.426, -0.381] | 0/14 of 14 |
| token | all | 256 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.424 [-0.443, -0.402] | 0/14 of 14 |
| token | all | 256 | ark:cuboid_mean:1 | score | -0.250 [-0.266, -0.233] | 0/14 of 14 |
| token | all | 256 | ema_z(lodo) | score | +0.180 [+0.161, +0.204] | 14/0 of 14 |
| token | all | 256 | ema_z(lodo) | lru | +0.232 [+0.219, +0.246] | 14/0 of 14 |
| token | all | 256 | ark:cuboid_mean:1 | arkvale_simple | -0.250 [-0.266, -0.233] | 0/14 of 14 |
| token | all | 256 | ark:digest(lodo):1 | ema_z(lodo) | -0.364 [-0.398, -0.337] | 0/14 of 14 |
| token | all | 256 | ark:quest:1 | ema_z(lodo) | -0.364 [-0.398, -0.337] | 0/14 of 14 |
| token | all | 256 | ark:cuboid_mean:1 | ark:quest:1 | -0.065 [-0.083, -0.044] | 1/13 of 14 |
| token | all | 256 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.187 [-0.202, -0.172] | 0/14 of 14 |
| token | a | 72 | ark:cuboid_mean:1 | ema_z(lodo) | -0.619 [-0.633, -0.606] | 0/14 of 14 |
| token | a | 72 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.615 [-0.629, -0.601] | 0/14 of 14 |
| token | a | 72 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.646 [-0.660, -0.633] | 0/14 of 14 |
| token | a | 72 | ark:cuboid_mean:1 | score | -0.591 [-0.605, -0.578] | 0/14 of 14 |
| token | a | 72 | ema_z(lodo) | score | +0.028 [+0.026, +0.031] | 14/0 of 14 |
| token | a | 72 | ema_z(lodo) | lru | +0.254 [+0.252, +0.257] | 14/0 of 14 |
| token | a | 72 | ark:cuboid_mean:1 | arkvale_simple | -0.591 [-0.605, -0.578] | 0/14 of 14 |
| token | a | 72 | ark:digest(lodo):1 | ema_z(lodo) | -0.066 [-0.068, -0.064] | 0/14 of 14 |
| token | a | 72 | ark:quest:1 | ema_z(lodo) | -0.066 [-0.068, -0.064] | 0/14 of 14 |
| token | a | 72 | ark:cuboid_mean:1 | ark:quest:1 | -0.553 [-0.567, -0.540] | 0/14 of 14 |
| token | a | 72 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.034 [-0.036, -0.032] | 0/14 of 14 |
| token | a | 80 | ark:cuboid_mean:1 | ema_z(lodo) | -0.708 [-0.729, -0.688] | 0/14 of 14 |
| token | a | 80 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.702 [-0.723, -0.682] | 0/14 of 14 |
| token | a | 80 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.727 [-0.747, -0.707] | 0/14 of 14 |
| token | a | 80 | ark:cuboid_mean:1 | score | -0.676 [-0.696, -0.656] | 0/14 of 14 |
| token | a | 80 | ema_z(lodo) | score | +0.032 [+0.031, +0.034] | 14/0 of 14 |
| token | a | 80 | ema_z(lodo) | lru | +0.273 [+0.269, +0.277] | 14/0 of 14 |
| token | a | 80 | ark:cuboid_mean:1 | arkvale_simple | -0.676 [-0.696, -0.656] | 0/14 of 14 |
| token | a | 80 | ark:digest(lodo):1 | ema_z(lodo) | -0.071 [-0.072, -0.069] | 0/14 of 14 |
| token | a | 80 | ark:quest:1 | ema_z(lodo) | -0.071 [-0.072, -0.069] | 0/14 of 14 |
| token | a | 80 | ark:cuboid_mean:1 | ark:quest:1 | -0.637 [-0.657, -0.618] | 0/14 of 14 |
| token | a | 80 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.032 [-0.035, -0.030] | 0/14 of 14 |
| token | a | 96 | ark:cuboid_mean:1 | ema_z(lodo) | -0.618 [-0.642, -0.594] | 0/14 of 14 |
| token | a | 96 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.610 [-0.634, -0.585] | 0/14 of 14 |
| token | a | 96 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.657 [-0.679, -0.634] | 0/14 of 14 |
| token | a | 96 | ark:cuboid_mean:1 | score | -0.570 [-0.595, -0.545] | 0/14 of 14 |
| token | a | 96 | ema_z(lodo) | score | +0.048 [+0.044, +0.051] | 14/0 of 14 |
| token | a | 96 | ema_z(lodo) | lru | +0.273 [+0.268, +0.276] | 14/0 of 14 |
| token | a | 96 | ark:cuboid_mean:1 | arkvale_simple | -0.570 [-0.595, -0.545] | 0/14 of 14 |
| token | a | 96 | ark:digest(lodo):1 | ema_z(lodo) | -0.097 [-0.100, -0.093] | 0/14 of 14 |
| token | a | 96 | ark:quest:1 | ema_z(lodo) | -0.097 [-0.100, -0.093] | 0/14 of 14 |
| token | a | 96 | ark:cuboid_mean:1 | ark:quest:1 | -0.521 [-0.545, -0.496] | 0/14 of 14 |
| token | a | 96 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.043 [-0.046, -0.040] | 0/14 of 14 |
| token | a | 128 | ark:cuboid_mean:1 | ema_z(lodo) | -0.439 [-0.460, -0.416] | 0/14 of 14 |
| token | a | 128 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.425 [-0.446, -0.401] | 0/14 of 14 |
| token | a | 128 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.468 [-0.488, -0.447] | 0/14 of 14 |
| token | a | 128 | ark:cuboid_mean:1 | score | -0.357 [-0.379, -0.334] | 0/14 of 14 |
| token | a | 128 | ema_z(lodo) | score | +0.082 [+0.077, +0.087] | 14/0 of 14 |
| token | a | 128 | ema_z(lodo) | lru | +0.256 [+0.250, +0.261] | 14/0 of 14 |
| token | a | 128 | ark:cuboid_mean:1 | arkvale_simple | -0.357 [-0.379, -0.334] | 0/14 of 14 |
| token | a | 128 | ark:digest(lodo):1 | ema_z(lodo) | -0.157 [-0.163, -0.151] | 0/14 of 14 |
| token | a | 128 | ark:quest:1 | ema_z(lodo) | -0.157 [-0.163, -0.151] | 0/14 of 14 |
| token | a | 128 | ark:cuboid_mean:1 | ark:quest:1 | -0.282 [-0.302, -0.259] | 0/14 of 14 |
| token | a | 128 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.072 [-0.077, -0.068] | 0/14 of 14 |
| token | a | 256 | ark:cuboid_mean:1 | ema_z(lodo) | -0.423 [-0.445, -0.398] | 0/14 of 14 |
| token | a | 256 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.398 [-0.419, -0.374] | 0/14 of 14 |
| token | a | 256 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.417 [-0.436, -0.395] | 0/14 of 14 |
| token | a | 256 | ark:cuboid_mean:1 | score | -0.242 [-0.259, -0.225] | 0/14 of 14 |
| token | a | 256 | ema_z(lodo) | score | +0.181 [+0.161, +0.205] | 14/0 of 14 |
| token | a | 256 | ema_z(lodo) | lru | +0.230 [+0.216, +0.243] | 14/0 of 14 |
| token | a | 256 | ark:cuboid_mean:1 | arkvale_simple | -0.242 [-0.259, -0.225] | 0/14 of 14 |
| token | a | 256 | ark:digest(lodo):1 | ema_z(lodo) | -0.358 [-0.391, -0.330] | 0/14 of 14 |
| token | a | 256 | ark:quest:1 | ema_z(lodo) | -0.358 [-0.391, -0.330] | 0/14 of 14 |
| token | a | 256 | ark:cuboid_mean:1 | ark:quest:1 | -0.065 [-0.083, -0.043] | 1/13 of 14 |
| token | a | 256 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.180 [-0.195, -0.165] | 0/14 of 14 |
| token | b | 128 | ark:cuboid_mean:1 | ema_z(lodo) | -0.653 [-0.682, -0.622] | 0/6 of 6 |
| token | b | 128 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.648 [-0.677, -0.618] | 0/6 of 6 |
| token | b | 128 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.675 [-0.703, -0.648] | 0/6 of 6 |
| token | b | 128 | ark:cuboid_mean:1 | score | -0.622 [-0.655, -0.589] | 0/6 of 6 |
| token | b | 128 | ema_z(lodo) | score | +0.031 [+0.027, +0.035] | 6/0 of 6 |
| token | b | 128 | ema_z(lodo) | lru | +0.266 [+0.255, +0.277] | 6/0 of 6 |
| token | b | 128 | ark:cuboid_mean:1 | arkvale_simple | -0.622 [-0.655, -0.589] | 0/6 of 6 |
| token | b | 128 | ark:digest(lodo):1 | ema_z(lodo) | -0.072 [-0.078, -0.067] | 0/6 of 6 |
| token | b | 128 | ark:quest:1 | ema_z(lodo) | -0.072 [-0.078, -0.067] | 0/6 of 6 |
| token | b | 128 | ark:cuboid_mean:1 | ark:quest:1 | -0.582 [-0.615, -0.549] | 0/6 of 6 |
| token | b | 128 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.036 [-0.041, -0.032] | 0/6 of 6 |
| token | b | 192 | ark:cuboid_mean:1 | ema_z(lodo) | -0.465 [-0.513, -0.416] | 0/14 of 14 |
| token | b | 192 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.453 [-0.501, -0.405] | 0/14 of 14 |
| token | b | 192 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.499 [-0.551, -0.446] | 0/14 of 14 |
| token | b | 192 | ark:cuboid_mean:1 | score | -0.388 [-0.446, -0.328] | 0/14 of 14 |
| token | b | 192 | ema_z(lodo) | score | +0.077 [+0.065, +0.090] | 14/0 of 14 |
| token | b | 192 | ema_z(lodo) | lru | +0.250 [+0.241, +0.258] | 14/0 of 14 |
| token | b | 192 | ark:cuboid_mean:1 | arkvale_simple | -0.388 [-0.446, -0.328] | 0/14 of 14 |
| token | b | 192 | ark:digest(lodo):1 | ema_z(lodo) | -0.151 [-0.175, -0.129] | 0/14 of 14 |
| token | b | 192 | ark:quest:1 | ema_z(lodo) | -0.151 [-0.175, -0.129] | 0/14 of 14 |
| token | b | 192 | ark:cuboid_mean:1 | ark:quest:1 | -0.314 [-0.382, -0.244] | 0/14 of 14 |
| token | b | 192 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.070 [-0.083, -0.058] | 0/14 of 14 |
| token | b | 256 | ark:cuboid_mean:1 | ema_z(lodo) | -0.398 [-0.420, -0.378] | 0/14 of 14 |
| token | b | 256 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.379 [-0.399, -0.359] | 0/14 of 14 |
| token | b | 256 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.411 [-0.429, -0.391] | 0/14 of 14 |
| token | b | 256 | ark:cuboid_mean:1 | score | -0.267 [-0.284, -0.246] | 0/14 of 14 |
| token | b | 256 | ema_z(lodo) | score | +0.131 [+0.116, +0.151] | 14/0 of 14 |
| token | b | 256 | ema_z(lodo) | lru | +0.228 [+0.216, +0.240] | 14/0 of 14 |
| token | b | 256 | ark:cuboid_mean:1 | arkvale_simple | -0.267 [-0.284, -0.246] | 0/14 of 14 |
| token | b | 256 | ark:digest(lodo):1 | ema_z(lodo) | -0.265 [-0.301, -0.234] | 0/14 of 14 |
| token | b | 256 | ark:quest:1 | ema_z(lodo) | -0.265 [-0.301, -0.234] | 0/14 of 14 |
| token | b | 256 | ark:cuboid_mean:1 | ark:quest:1 | -0.133 [-0.159, -0.102] | 1/13 of 14 |
| token | b | 256 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.133 [-0.152, -0.117] | 0/14 of 14 |
| token | b | 384 | ark:cuboid_mean:1 | ema_z(lodo) | -0.461 [-0.491, -0.435] | 0/14 of 14 |
| token | b | 384 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.442 [-0.476, -0.415] | 0/14 of 14 |
| token | b | 384 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.450 [-0.484, -0.422] | 0/14 of 14 |
| token | b | 384 | ark:cuboid_mean:1 | score | -0.264 [-0.296, -0.236] | 0/14 of 14 |
| token | b | 384 | ema_z(lodo) | score | +0.197 [+0.180, +0.214] | 14/0 of 14 |
| token | b | 384 | ema_z(lodo) | lru | +0.232 [+0.212, +0.254] | 14/0 of 14 |
| token | b | 384 | ark:cuboid_mean:1 | arkvale_simple | -0.264 [-0.296, -0.236] | 0/14 of 14 |
| token | b | 384 | ark:digest(lodo):1 | ema_z(lodo) | -0.427 [-0.462, -0.400] | 0/14 of 14 |
| token | b | 384 | ark:quest:1 | ema_z(lodo) | -0.427 [-0.462, -0.400] | 0/14 of 14 |
| token | b | 384 | ark:cuboid_mean:1 | ark:quest:1 | -0.034 [-0.051, -0.016] | 3/11 of 14 |
| token | b | 384 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.233 [-0.265, -0.207] | 0/14 of 14 |
| token | b | 512 | ark:cuboid_mean:1 | ema_z(lodo) | -0.536 [-0.601, -0.492] | 0/14 of 14 |
| token | b | 512 | ark:cuboid_mean:1 | ema_allrank(lodo) | -0.536 [-0.603, -0.492] | 0/14 of 14 |
| token | b | 512 | ark:cuboid_mean:1 | powerlaw(lodo) | -0.518 [-0.589, -0.469] | 0/14 of 14 |
| token | b | 512 | ark:cuboid_mean:1 | score | -0.355 [-0.427, -0.307] | 0/14 of 14 |
| token | b | 512 | ema_z(lodo) | score | +0.181 [+0.136, +0.219] | 14/0 of 14 |
| token | b | 512 | ema_z(lodo) | lru | +0.341 [+0.296, +0.399] | 14/0 of 14 |
| token | b | 512 | ark:cuboid_mean:1 | arkvale_simple | -0.355 [-0.427, -0.307] | 0/14 of 14 |
| token | b | 512 | ark:digest(lodo):1 | ema_z(lodo) | -0.536 [-0.601, -0.492] | 0/14 of 14 |
| token | b | 512 | ark:quest:1 | ema_z(lodo) | -0.542 [-0.606, -0.500] | 0/14 of 14 |
| token | b | 512 | ark:cuboid_mean:1 | ark:quest:1 | +0.006 [-0.017, +0.028] | 8/6 of 14 |
| token | b | 512 | ark:cuboid_mean:1 | ark:cuboid_mean:0 | -0.366 [-0.433, -0.317] | 0/14 of 14 |

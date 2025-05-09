import numpy as np

jam_sample_actions = [
    np.array([92.0, 91.0, 0.0], dtype=np.float32),
    np.array([92.0, 91.0, 0.0], dtype=np.float32),
    np.array([92.0, 91.0, 0.0], dtype=np.float32),
    np.array([175.0, 88.0, 0.0], dtype=np.float32),
    np.array([225.0, 85.0, 0.0], dtype=np.float32),
    np.array([257.0, 87.0, 0.0], dtype=np.float32),
    np.array([290.0, 87.0, 0.0], dtype=np.float32),
    np.array([324.0, 83.0, 0.0], dtype=np.float32),
    np.array([365.0, 80.0, 0.0], dtype=np.float32),
    np.array([412.0, 69.0, 0.0], dtype=np.float32),
    np.array([444.0, 67.0, 0.0], dtype=np.float32),
    np.array([474.0, 67.0, 0.0], dtype=np.float32),
    np.array([499.0, 67.0, 0.0], dtype=np.float32),
    np.array([517.0, 69.0, 0.0], dtype=np.float32),
    np.array([533.0, 72.0, 0.0], dtype=np.float32),
    np.array([547.0, 74.0, 0.0], dtype=np.float32),
    np.array([556.0, 75.0, 0.0], dtype=np.float32),
    np.array([567.0, 75.0, 0.0], dtype=np.float32),
    np.array([570.0, 76.0, 0.0], dtype=np.float32),
    np.array([574.0, 77.0, 0.0], dtype=np.float32),
    np.array([575.0, 77.0, 0.5], dtype=np.float32),
    np.array([575.0, 78.0, 0.5], dtype=np.float32),
    np.array([575.0, 78.0, 0.5], dtype=np.float32),
    np.array([575.0, 78.0, 0.5], dtype=np.float32),
    np.array([575.0, 78.0, 0.5], dtype=np.float32),
    np.array([507.0, 95.0, 0.5], dtype=np.float32),
    np.array([446.0, 124.0, 0.5], dtype=np.float32),
    np.array([388.0, 160.0, 0.5], dtype=np.float32),
    np.array([366.0, 176.0, 0.5], dtype=np.float32),
    np.array([348.0, 190.0, 0.5], dtype=np.float32),
    np.array([299.0, 232.0, 0.5], dtype=np.float32),
    np.array([230.0, 316.0, 0.5], dtype=np.float32),
    np.array([210.0, 359.0, 0.5], dtype=np.float32),
    np.array([197.0, 395.0, 0.5], dtype=np.float32),
    np.array([180.0, 435.0, 0.5], dtype=np.float32),
    np.array([179.0, 437.0, 0.5], dtype=np.float32),
    np.array([179.0, 438.0, 0.5], dtype=np.float32),
    np.array([166.0, 472.0, 0.5], dtype=np.float32),
    np.array([165.0, 490.0, 0.5], dtype=np.float32),
    np.array([163.0, 510.0, 0.5], dtype=np.float32),
    np.array([159.0, 527.0, 0.5], dtype=np.float32),
    np.array([159.0, 530.0, 0.5], dtype=np.float32),
    np.array([159.0, 530.0, 0.5], dtype=np.float32),
    np.array([159.0, 530.0, 0.5], dtype=np.float32),
    np.array([160.0, 526.0, 0.5], dtype=np.float32),
    np.array([160.0, 525.0, 0.5], dtype=np.float32),
    np.array([160.0, 525.0, 0.5], dtype=np.float32),
    np.array([160.0, 525.0, 0.5], dtype=np.float32),
    np.array([160.0, 525.0, 1.0], dtype=np.float32),
    np.array([160.0, 524.0, 1.0], dtype=np.float32),
    np.array([160.0, 524.0, 1.0], dtype=np.float32),
    np.array([160.0, 524.0, 1.0], dtype=np.float32),
    np.array([159.0, 486.0, 1.0], dtype=np.float32),
    np.array([157.0, 462.0, 1.0], dtype=np.float32),
    np.array([156.0, 446.0, 1.0], dtype=np.float32),
    np.array([156.0, 432.0, 1.0], dtype=np.float32),
    np.array([157.0, 420.0, 1.0], dtype=np.float32),
    np.array([159.0, 399.0, 1.0], dtype=np.float32),
    np.array([161.0, 373.0, 1.0], dtype=np.float32),
    np.array([165.0, 335.0, 1.0], dtype=np.float32),
    np.array([166.0, 318.0, 1.0], dtype=np.float32),
    np.array([166.0, 302.0, 1.0], dtype=np.float32),
    np.array([166.0, 296.0, 1.0], dtype=np.float32),
    np.array([166.0, 296.0, 1.0], dtype=np.float32),
    np.array([166.0, 296.0, 1.0], dtype=np.float32),
    np.array([164.0, 252.0, 1.0], dtype=np.float32),
    np.array([164.0, 228.0, 1.0], dtype=np.float32),
    np.array([165.0, 212.0, 1.0], dtype=np.float32),
    np.array([165.0, 210.0, 1.0], dtype=np.float32),
    np.array([168.0, 198.0, 1.0], dtype=np.float32),
    np.array([169.0, 189.0, 1.0], dtype=np.float32),
    np.array([172.0, 188.0, 1.0], dtype=np.float32),
    np.array([201.0, 191.0, 1.0], dtype=np.float32),
    np.array([229.0, 193.0, 1.0], dtype=np.float32),
    np.array([273.0, 198.0, 1.0], dtype=np.float32),
    np.array([331.0, 196.0, 1.0], dtype=np.float32),
    np.array([367.0, 194.0, 1.0], dtype=np.float32),
    np.array([393.0, 194.0, 1.0], dtype=np.float32),
    np.array([417.0, 196.0, 1.0], dtype=np.float32),
    np.array([441.0, 196.0, 1.0], dtype=np.float32),
    np.array([464.0, 196.0, 1.0], dtype=np.float32),
    np.array([475.0, 196.0, 1.0], dtype=np.float32),
    np.array([475.0, 196.0, 1.0], dtype=np.float32),
    np.array([475.0, 199.0, 1.0], dtype=np.float32),
    np.array([475.0, 203.0, 1.0], dtype=np.float32),
    np.array([464.0, 250.0, 1.0], dtype=np.float32),
    np.array([454.0, 310.0, 1.0], dtype=np.float32),
    np.array([451.0, 363.0, 1.0], dtype=np.float32),
    np.array([448.0, 398.0, 1.0], dtype=np.float32),
    np.array([448.0, 422.0, 1.0], dtype=np.float32),
    np.array([447.0, 447.0, 1.0], dtype=np.float32),
    np.array([447.0, 462.0, 1.0], dtype=np.float32),
    np.array([447.0, 490.0, 1.0], dtype=np.float32),
    np.array([447.0, 493.0, 1.0], dtype=np.float32),
    np.array([448.0, 493.0, 1.0], dtype=np.float32),
    np.array([448.0, 493.0, 1.0], dtype=np.float32),
    np.array([448.0, 493.0, 1.0], dtype=np.float32),
    np.array([448.0, 493.0, 1.0], dtype=np.float32),
    np.array([422.0, 493.0, 1.0], dtype=np.float32),
    np.array([356.0, 498.0, 1.0], dtype=np.float32),
    np.array([315.0, 505.0, 1.0], dtype=np.float32),
    np.array([274.0, 511.0, 1.0], dtype=np.float32),
    np.array([241.0, 514.0, 1.0], dtype=np.float32),
    np.array([208.0, 520.0, 1.0], dtype=np.float32),
    np.array([182.0, 521.0, 1.0], dtype=np.float32),
    np.array([160.0, 521.0, 1.0], dtype=np.float32),
    np.array([149.0, 519.0, 1.0], dtype=np.float32),
    np.array([146.0, 518.0, 1.0], dtype=np.float32),
    np.array([144.0, 518.0, 1.0], dtype=np.float32),
    np.array([144.0, 517.0, 0.5], dtype=np.float32),
    np.array([144.0, 517.0, 0.5], dtype=np.float32),
    np.array([144.0, 517.0, 0.5], dtype=np.float32),
    np.array([144.0, 517.0, 0.5], dtype=np.float32),
    np.array([151.0, 516.0, 0.5], dtype=np.float32),
    np.array([153.0, 514.0, 0.5], dtype=np.float32),
    np.array([163.0, 496.0, 0.5], dtype=np.float32),
    np.array([266.0, 343.0, 0.5], dtype=np.float32),
    np.array([321.0, 289.0, 0.5], dtype=np.float32),
    np.array([362.0, 252.0, 0.5], dtype=np.float32),
    np.array([401.0, 218.0, 0.5], dtype=np.float32),
    np.array([447.0, 186.0, 0.5], dtype=np.float32),
    np.array([486.0, 162.0, 0.5], dtype=np.float32),
    np.array([520.0, 139.0, 0.5], dtype=np.float32),
    np.array([538.0, 128.0, 0.5], dtype=np.float32),
    np.array([551.0, 121.0, 0.5], dtype=np.float32),
    np.array([558.0, 115.0, 0.5], dtype=np.float32),
    np.array([562.0, 112.0, 0.5], dtype=np.float32),
    np.array([566.0, 106.0, 0.5], dtype=np.float32),
    np.array([569.0, 101.0, 0.5], dtype=np.float32),
    np.array([571.0, 97.0, 0.5], dtype=np.float32),
    np.array([572.0, 87.0, 0.5], dtype=np.float32),
    np.array([572.0, 86.0, 0.5], dtype=np.float32),
    np.array([572.0, 86.0, 0.5], dtype=np.float32),
    np.array([572.0, 86.0, 0.0], dtype=np.float32),
]

import numpy as np

jam_sample_actions = [
    np.array([91.0, 85.0, 0.0], dtype=np.float32),
    np.array([91.0, 85.0, 0.0], dtype=np.float32),
    np.array([91.0, 85.0, 0.0], dtype=np.float32),
    np.array([132.0, 85.0, 0.0], dtype=np.float32),
    np.array([242.0, 96.0, 0.0], dtype=np.float32),
    np.array([299.0, 97.0, 0.0], dtype=np.float32),
    np.array([348.0, 96.0, 0.0], dtype=np.float32),
    np.array([390.0, 94.0, 0.0], dtype=np.float32),
    np.array([430.0, 91.0, 0.0], dtype=np.float32),
    np.array([465.0, 89.0, 0.0], dtype=np.float32),
    np.array([493.0, 87.0, 0.0], dtype=np.float32),
    np.array([513.0, 86.0, 0.0], dtype=np.float32),
    np.array([530.0, 86.0, 0.0], dtype=np.float32),
    np.array([555.0, 89.0, 0.0], dtype=np.float32),
    np.array([568.0, 89.0, 0.0], dtype=np.float32),
    np.array([575.0, 90.0, 0.0], dtype=np.float32),
    np.array([580.0, 90.0, 0.5], dtype=np.float32),
    np.array([580.0, 91.0, 0.5], dtype=np.float32),
    np.array([580.0, 91.0, 0.5], dtype=np.float32),
    np.array([580.0, 91.0, 0.5], dtype=np.float32),
    np.array([546.0, 108.0, 0.5], dtype=np.float32),
    np.array([435.0, 193.0, 0.5], dtype=np.float32),
    np.array([356.0, 261.0, 0.5], dtype=np.float32),
    np.array([328.0, 282.0, 0.5], dtype=np.float32),
    np.array([312.0, 301.0, 0.5], dtype=np.float32),
    np.array([297.0, 317.0, 0.5], dtype=np.float32),
    np.array([293.0, 325.0, 0.5], dtype=np.float32),
    np.array([289.0, 329.0, 0.5], dtype=np.float32),
    np.array([283.0, 336.0, 0.5], dtype=np.float32),
    np.array([280.0, 338.0, 0.5], dtype=np.float32),
    np.array([281.0, 336.0, 0.5], dtype=np.float32),
    np.array([281.0, 336.0, 0.5], dtype=np.float32),
    np.array([279.0, 341.0, 0.5], dtype=np.float32),
    np.array([279.0, 341.0, 1.0], dtype=np.float32),
    np.array([279.0, 341.0, 1.0], dtype=np.float32),
    np.array([279.0, 341.0, 1.0], dtype=np.float32),
    np.array([279.0, 341.0, 1.0], dtype=np.float32),
    np.array([268.0, 302.0, 1.0], dtype=np.float32),
    np.array([268.0, 292.0, 1.0], dtype=np.float32),
    np.array([273.0, 286.0, 1.0], dtype=np.float32),
    np.array([282.0, 281.0, 1.0], dtype=np.float32),
    np.array([296.0, 277.0, 1.0], dtype=np.float32),
    np.array([313.0, 274.0, 1.0], dtype=np.float32),
    np.array([331.0, 274.0, 1.0], dtype=np.float32),
    np.array([353.0, 287.0, 1.0], dtype=np.float32),
    np.array([366.0, 302.0, 1.0], dtype=np.float32),
    np.array([375.0, 325.0, 1.0], dtype=np.float32),
    np.array([380.0, 352.0, 1.0], dtype=np.float32),
    np.array([368.0, 379.0, 1.0], dtype=np.float32),
    np.array([354.0, 402.0, 1.0], dtype=np.float32),
    np.array([332.0, 422.0, 1.0], dtype=np.float32),
    np.array([302.0, 432.0, 1.0], dtype=np.float32),
    np.array([265.0, 433.0, 1.0], dtype=np.float32),
    np.array([234.0, 426.0, 1.0], dtype=np.float32),
    np.array([212.0, 410.0, 1.0], dtype=np.float32),
    np.array([193.0, 384.0, 1.0], dtype=np.float32),
    np.array([184.0, 362.0, 1.0], dtype=np.float32),
    np.array([179.0, 335.0, 1.0], dtype=np.float32),
    np.array([181.0, 310.0, 1.0], dtype=np.float32),
    np.array([192.0, 278.0, 1.0], dtype=np.float32),
    np.array([211.0, 250.0, 1.0], dtype=np.float32),
    np.array([237.0, 219.0, 1.0], dtype=np.float32),
    np.array([261.0, 200.0, 1.0], dtype=np.float32),
    np.array([289.0, 192.0, 1.0], dtype=np.float32),
    np.array([323.0, 192.0, 1.0], dtype=np.float32),
    np.array([353.0, 195.0, 1.0], dtype=np.float32),
    np.array([406.0, 218.0, 1.0], dtype=np.float32),
    np.array([435.0, 237.0, 1.0], dtype=np.float32),
    np.array([451.0, 254.0, 1.0], dtype=np.float32),
    np.array([460.0, 272.0, 1.0], dtype=np.float32),
    np.array([460.0, 315.0, 1.0], dtype=np.float32),
    np.array([449.0, 362.0, 1.0], dtype=np.float32),
    np.array([440.0, 402.0, 1.0], dtype=np.float32),
    np.array([432.0, 428.0, 1.0], dtype=np.float32),
    np.array([417.0, 456.0, 1.0], dtype=np.float32),
    np.array([402.0, 471.0, 1.0], dtype=np.float32),
    np.array([383.0, 479.0, 1.0], dtype=np.float32),
    np.array([367.0, 487.0, 1.0], dtype=np.float32),
    np.array([339.0, 496.0, 1.0], dtype=np.float32),
    np.array([316.0, 500.0, 1.0], dtype=np.float32),
    np.array([299.0, 501.0, 1.0], dtype=np.float32),
    np.array([272.0, 503.0, 1.0], dtype=np.float32),
    np.array([239.0, 505.0, 1.0], dtype=np.float32),
    np.array([214.0, 505.0, 1.0], dtype=np.float32),
    np.array([196.0, 506.0, 1.0], dtype=np.float32),
    np.array([179.0, 506.0, 1.0], dtype=np.float32),
    np.array([167.0, 505.0, 1.0], dtype=np.float32),
    np.array([166.0, 505.0, 1.0], dtype=np.float32),
    np.array([165.0, 505.0, 1.0], dtype=np.float32),
    np.array([165.0, 505.0, 0.5], dtype=np.float32),
    np.array([165.0, 505.0, 0.5], dtype=np.float32),
    np.array([165.0, 505.0, 0.5], dtype=np.float32),
    np.array([165.0, 505.0, 0.5], dtype=np.float32),
    np.array([165.0, 505.0, 0.5], dtype=np.float32),
    np.array([183.0, 448.0, 0.5], dtype=np.float32),
    np.array([201.0, 401.0, 0.5], dtype=np.float32),
    np.array([225.0, 345.0, 0.5], dtype=np.float32),
    np.array([271.0, 276.0, 0.5], dtype=np.float32),
    np.array([346.0, 190.0, 0.5], dtype=np.float32),
    np.array([394.0, 151.0, 0.5], dtype=np.float32),
    np.array([419.0, 138.0, 0.5], dtype=np.float32),
    np.array([446.0, 129.0, 0.5], dtype=np.float32),
    np.array([473.0, 123.0, 0.5], dtype=np.float32),
    np.array([491.0, 116.0, 0.5], dtype=np.float32),
    np.array([509.0, 111.0, 0.5], dtype=np.float32),
    np.array([526.0, 107.0, 0.5], dtype=np.float32),
    np.array([540.0, 106.0, 0.5], dtype=np.float32),
    np.array([551.0, 105.0, 0.5], dtype=np.float32),
    np.array([558.0, 104.0, 0.5], dtype=np.float32),
    np.array([565.0, 102.0, 0.5], dtype=np.float32),
    np.array([569.0, 101.0, 0.5], dtype=np.float32),
    np.array([571.0, 95.0, 0.5], dtype=np.float32),
    np.array([573.0, 92.0, 0.5], dtype=np.float32),
    np.array([573.0, 91.0, 0.5], dtype=np.float32),
    np.array([573.0, 91.0, 0.0], dtype=np.float32),
]

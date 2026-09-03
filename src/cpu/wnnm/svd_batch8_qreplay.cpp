#include <atomic>

#define NSS_SVD_QREPLAY_ROW_MAJOR 1
#define SvdEconomy8Batch SvdEconomy8BatchQReplay
#define SvdEconomy8BatchU SvdEconomy8BatchUQReplay
#define svd_economy_8_batch_hwy svd_economy_8_batch_qreplay_hwy
#define svd_economy_8_batch_u_hwy svd_economy_8_batch_u_qreplay_hwy

#include "cpu/wnnm/svd_batch8.cpp"

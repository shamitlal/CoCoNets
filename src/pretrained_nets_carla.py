total_init = ""
feat2D_init = ""
feat3D_init = ""
feat3docc_init = ""
view_init = ""
flow_init = ""
emb2D_init = ""
vis_init = ""
occ_init = ""
resolve_init = ""
ego_init = ""
det_init = ""
forecast_init = ""
preocc_init = ""
vqrgb_init = ""
optim_init = ""
localdecoder_init = ""
localdecoder_render_init = ""
# some coeffs
emb_dim = 8
feat3D_dim = 32
feat2D_dim = 32
view_depth = 32

ckpt = "02_s6_m128x8x128_p64x192_1e-4_F64f_V3rf_n512_S3i_c1_Vf_d64_mabs7i3t_mabs7i3v_earth39"
ckpt = "02_s5_m128x8x128_p64x192_1e-4_F64f_V3rf_n512_S3i_c1_Vf_d64_mabs7i3t_mabs7i3v_earth38"
ckpt = "02_s2_m128x8x128_p64x192_1e-4_F64_V3r_n512_l1_O_c.1_s.001_V_d64_e1_mabs7i3t_mabs7i3v_r44"
ckpt = "02_s2_m128x8x128_p64x192_1e-4_F64_s.01_V3r_n512_l1_O_c.1_s.001_V_d64_e1_mabs7i3t_mabs7i3v_r46"
ckpt = "02_s4_m128x8x128_p64x192_1e-4_F64f_V3rf_n512_S3i_c1_Of_Vf_d64_mabs7i3t_mabs7i3v_radial04"
ckpt = "02_s4_m128x8x128_p64x192_1e-4_F64f_V3rf_n512_S3i_c1_Vf_d64_mabs7i3t_mabs7i3v_earth45"
ckpt = "02_s5_m128x8x128_p64x192_1e-4_F64f_V3rf_n512_S3i_c1_Vf_d64_mabs7i3t_mabs7i3v_earth38"
ckpt = "02_s2_m128x8x128_p64x192_1e-4_F64_s.01_V3r_n512_l1_O_c.1_s.001_V_d64_e1_mabs7i3t_mabs7i3v_r46"
ckpt = "02_s5_m128x8x128_p64x192_1e-4_F64f_V3rf_n512_S3i_c1_Vf_d64_mabs7i3t_mabs7i3v_earth38"
ckpt = "02_s4_m128x8x128_p64x192_1e-4_F64f_V3rf_n512_S3i_c1_Of_Vf_d64_mabs7i3t_mabs7i3v_radial11"
ckpt = "02_s2_m128x8x128_p64x192_1e-4_F64_s.01_V3r_n512_l1_O_c.1_s.001_V_d64_e1_mabs7i3t_mabs7i3v_r46"
ckpt = "02_s3_m128x8x128_p64x192_1e-4_F2_d64_F3_d64_V3r_n512_l2_V_d64_e.1_E2_s1_m.1_e2_n32_E3_m1_e.1_n2_d16_L_c1_mabs7i3t_vq13"
ckpt = "02_s3_m128x8x128_p64x192_1e-4_F2_d64_F3_d64_V3r_n512_l2_O_c.1_V_d64_e.1_E2_s1_m.1_e2_n32_E3_m1_e.1_n2_d16_L_c1_mabs7i3t_vq14"

ckpt = "04_m128x32x128_p64x192_1e-3_F32_V3r_l1_V_d32_c1_mabs7i3t_25"
ckpt = "04_m128x32x128_p64x192_1e-3_F32_V3r_l1_V_d32_c1_mabs7i3t_24"
ckpt = "04_m64x16x64_1e-3_Vr_r1_l1_mabs7i3t_11"

ckpt = "02_s3_m128x8x128_p64x192_1e-4_F2_d64_F3_d64_V3r_n512_l2_O_c.1_V_d64_e.1_E2_s1_m.1_e2_n32_E3_m1_e.1_n2_d16_L_c1_mabs7i3t_vq14"
ckpt = "02_s4_m128x8x128_p64x192_1e-4_F3f_d64_V3rf_n512_S3i_c1_Of_Vf_d64_mabs7i3t_mabs7i3v_go00"
ckpt = "02_s1_m128x32x128_1e-4_F3_d32_L_c10_fags16i3t_time52"
ckpt = "02_s1_m128x32x128_1e-4_F3_d32_L_c10_fags16i3t_time52"
ckpt = "02_s3_m128x16x128_p64x192_1e-4_F2_d64_F3_d64_V_d64_e.1_E2_s1_m.1_e2_n32_E3_m1_e.1_n2_d16_mabs7i3t_mabs7i3v_bench03"
# ckpt = "02_s4_m128x8x128_p64x192_1e-4_F3f_d64_V3rf_n512_G3v_c1_Of_Vf_d64_mabs7i3t_mabs7i3v_go02"
# ckpt = "04_m128x32x128_p64x192_1e-3_F32_V_d32_c1_mabs7i3t_23" #works

# ckpt = "02_s4_m128x8x128_p64x192_1e-4_F64f_V3rf_n512_S3i_c1_Of_Vf_d64_mabs7i3t_mabs7i3v_raidal12"

# ckpt = "02_s3_m128x16x128_1e-5_F3_d64_O_c.1_s.001_E3_n2_d16_c1_koacs10i2a_ktads8i1a_t05"
# ckpt = "02_s3_m128x16x128_1e-5_F3_d64_E3_n2_d16_c1_koacs10i2a_ktads8i1a_t09"
# ckpt = "02_s3_m128x16x128_1e-5_F3_d64_E3_n2_d16_c1_koacs10i2a_ktads8i1a_b17"
# ckpt = "02_s3_m128x16x128_1e-5_F3_d64_E3_n2_d16_c1_koads4i4a_ktads8i1a_b16"
# ckpt = "02_s3_m128x16x128_1e-5_F3_d64_O_c.1_s.001_E3_n2_d16_c1_koacs10i2a_ktads8i1a_b18"
# ckpt = "02_s2_1e-4_F2_d32_T2_e100_d1_mabs7i3t_don22"
# ckpt = "02_s2_1e-4_F2_d64_s.001_T2_e100_d1_mabs7i3t_don23"
# ckpt = "02_s2_1e-4_F2_d64_T2_e100_d1_koacs10i2a_ktads8i1a_test03"
# ckpt = "02_s2_1e-4_F2_d64_T2_e100_d1_mabs7i3t_don25"
# ckpt = "02_s2_m256x128x256_1e-4_F3_d16_M_c1_ktads8i1t_ktads8i1v_siam19"
# ckpt = "02_s2_1e-4_F2_d64_C2_h100_mabs7i3t_co00"
# ckpt = "02_s2_1e-4_F2_d32_s.01_C2_h100_koacs10i2a_ktads8i1a_col31"
# ckpt = "02_s2_m128x16x128_1e-4_F3_d64_E3_n2_d16_c1_koacs10i2a_ktads8i1a_g02"
# ckpt = "02_s2_m128x16x128_1e-4_F3_d64_E3_n2_d16_c1_koacs10i2a_ktads8i1a_g03"
ckpt = "02_s2_m128x16x128_1e-4_F3_d64_E3_n4_d16_c1_mabs7i3t_fags16i3v_moc15"
# ckpt = "02_s10_m256x128x256_z32x32x32_1e-4_F3_d32_M_c1_fags16i3t_fags16i3v_crazy10"
ckpt = "06_s2_m128x64x128_1e-4_F3_d64_R_r.1_t1_faks30i1t_faks30i1v_red48"
ckpt = "06_s2_m128x64x128_1e-5_F3_d64_R_r.1_t1_fags16i3t_fags16i3v_red52"
ckpt = "06_s2_m128x64x128_1e-4_F3_d64_Ro_c10_faks30i1one_faks30i1one_rob21"
ckpt = "02_s2_m128x8x128_p64x192_1e-4_F64_s.01_V3r_n512_l1_O_c.1_s.001_V_d64_e1_mabs7i3t_mabs7i3v_r46"
ckpt = "06_s2_m128x64x128_1e-5_F3_d64_Ri_r.1_t1_fags16i3t_faks30i1v_red54"
ckpt = "06_s2_m256x128x256_z48x48x48_1e-4_F3_d64_Ric2_r.1_t1_faks30i1t_faks30i1v_faks30i1v_rule27"
ckpt = "06_s2_m256x128x256_z48x48x48_1e-4_F3_d64_Ric2_r.1_t1_faks30i1t_faks30i1v_faks30i1v_rule27"
# ckpt = "06_s2_m192x96x192_z48x48x48_1e-4_F3_d64_Ric2_r.1_t1_faks30i1t_faks30i1v_faks30i1v_rule27"
ckpt = "01_s2_m256x128x256_z64x64x64_1e-3_F3_d64_M_c1_faks30i1t_faks30i1v_faks30i1v_cig28"
ckpt = "04_s2_m256x128x256_z64x64x64_1e-4_F3_d64_M_c1_r.1_faks30i1t_faks30i1v_faks30i1v_cig09"
ckpt = "04_s2_m256x128x256_z64x64x64_1e-4_F3_d64_M_c1_r.1_faks30i1t_faks30i1tce_faks30i1vce_mul20"
ckpt = "04_s2_m256x128x256_z64x64x64_1e-5_F3_d64_M_c1_r.1_faks30i1t_faks30i1tce_faks30i1vce_mul21"
# ckpt = "04_s60_m256x128x256_1e-5_F3_d32_Mr_t3_k8_c1_w.01_taes60i1t_taes60i1tce_taes60i1vce_f53"
# ckpt = "04_s60_m256x128x256_1e-5_F3_d32_Mr_t3_k8_c1_w.01_taes60i1t_taes60i1tce_taes60i1vce_f53"
# ckpt = "02_s60_m256x128x256_1e-5_F3_d32_Mr_t3_k8_e.1_w.0001_tafs60i1t_tafs60i1tce_tafs60i1vce_f56"
# ckpt = "02_s60_m256x128x256_1e-5_F3_d32_Mrd_p4_f56_k3_e.1_w.001_tafs60i1t_tafs60i1tce_tafs60i1vce_f69"
ckpt = "04_s2_m256x128x256_z64x64x64_1e-4_F3_d64_M_c1_r.1_Mrd_p4_f56_k5_e.1_w.001_vabs8i1t_tafs60i1tce_tags90i1vce_share33"
# ckpt = "02_s2_m256x128x256_z64x64x64_1e-5_F3_d64_M_c1_r.1_Mrd_p4_f56_k5_e.1_w.001_vabs8i1t_tags90i1t_tags90i1vce_share51"
ckpt = "02_s2_m256x128x256_z64x64x64_1e-5_F3_d64_M_c1_r.1_Mrd_p4_f56_k5_e.1_w.001_vabs8i1t_tags90i1t_tags90i1vce_share53"
ckpt = "02_s2_m256x128x256_z64x64x64_1e-5_F3_d64_M_c1_r.1_Mrd_p4_f56_k5_e.1_w.001_vabs8i1t_tags90i1t_tags90i1vce_share53"
ckpt = "04_s2_m256x128x256_z64x64x64_1e-4_F3_d64_M_c1_r.1_Mrd_p4_f56_k5_e1_w.001_s.001_vabs8i1t_tags90i1tb_tags90i1vce_share68"
ckpt = "04_s2_m256x128x256_z64x64x64_1e-5_F3_d64_M_c1_r.1_Mrd_p4_f56_k5_e1_w.001_s.01_vabs8i1t_tags90i1tb_tags90i1vce_share68"
ckpt = "04_s2_m256x128x256_z64x64x64_1e-3_F3f_d64_Mf_c1_r.1_Mrd_p4_f56_k5_e1_w.001_s.01_tags90i1tb_share74"
ckpt = "04_s2_m256x128x256_z64x64x64_1e-3_F3f_d64_Mf_c1_r.1_Mrd_p4_f56_k5_e1_w.001_s.01_tags90i1tb_share73"
ckpt = "04_s2_m256x128x256_z64x64x64_1e-3_F3f_d64_Mf_c1_r.1_Mrd_p4_f56_k5_e1_w.001_s.01_tags90i1tb_share74"
ckpt = "04_s2_m256x128x256_z64x64x64_1e-3_F3f_d64_Mf_c1_r.1_Mrd_p4_f56_k5_e1_w.001_s.01_tags90i1tb_share73"
ckpt = "04_s2_m256x128x256_z64x64x64_1e-3_F3f_d64_Mf_c1_r.1_Mrd_p4_f56_k5_e1_w.001_s.01_tags90i1tb_share80"
ckpt = "04_s2_m256x128x256_z64x64x64_1e-3_F3f_d64_Mf_c1_r.1_Mrd_p4_f56_k5_e1_w.001_s.01_tags90i1tb_share80"
ckpt = "04_s2_m256x128x256_z64x64x64_1e-3_F3f_d64_Mf_c1_r.1_Mrd_p4_f56_k5_e1_w.01_tags90i1tb_share81"
ckpt = "04_s2_m256x128x256_z64x64x64_1e-3_F3f_d64_Mf_c1_r.1_Mrd_p4_f56_k5_e1_w.01_tags90i1tb_share82"
# ckpt = "04_s2_m256x128x256_z64x64x64_1e-4_F3f_d64_Mf_c1_r.1_C_c1_vacs8i1t_vacs8i1v_conf26"
# ckpt = "04_s2_m256x128x256_z64x64x64_1e-4_F3f_d64_Mf_c1_r.1_C_c1_vacs8i1t_vacs8i1v_conf34"
# ckpt = "04_s2_m256x128x256_z64x64x64_1e-4_F3f_d64_Mf_c1_r.1_C_c1_vacs8i1t_vacs8i1v_conf35"
# ckpt = "04_s2_m256x128x256_z64x64x64_1e-4_F3f_d64_Mf_c1_r.1_C_c1_tals90i1t_tals90i1v_conf36"
ckpt = "04_s2_m256x128x256_z64x64x64_1e-4_F3f_d64_Mf_c1_r.1_C_c1_tals90i1t_tals90i1v_conf37"
ckpt = "02_s2_m256x128x256_z64x64x64_1e-3_F3f_d64_Mf_c1_r.1_Mrd_p4_f26_k5_e100_w.001_taps100i2tb_taps100i2vce_mat28"
ckpt = "02_s2_m256x128x256_z64x64x64_1e-3_F3f_d64_Mf_c1_r.1_Mrd_p4_f26_k5_e100_w.001_taps100i2tb_taps100i2vce_mat30"
ckpt = "04_s2_m256x128x256_z64x64x64_1e-3_F3f_d64_Mf_c1_r.1_Mrd_p4_f36_k5_e1_w.001_taps100i2tb_taps100i2vce_mat32"
ckpt = "04_s2_m256x128x256_z64x64x64_1e-3_F3f_d64_Mf_c1_r.1_Mrd_p4_f36_k5_e100_w.001_taps100i2tb_taps100i2vce_mat32"
ckpt = "01_s3_m160x160x160_1e-4_F3_d32_O_c1_mabs7i3t_bot78"
ckpt = "01_s7_m160x160x160_1e-4_F3_d32_O_c1_mabs7i3t_bot80"
ckpt = "01_s7_m160x160x160_1e-4_F3_d32_O_c1_C_c100_mabs7i3t_cen29"
ckpt = "01_s5_m160x160x160_1e-4_F3_d32_O_c1_C_p1_mabs7i3t_cen31"
ckpt = "01_s5_m160x160x160_1e-3_F3_d32_O_c1_C_p1_mabs7i3t_cen32"
ckpt = "01_s5_m160x160x160_1e-4_F3_d32_U_O_c1_C_p1_S_p1_mads7i3a_up00"
ckpt = "02_s5_m160x160x160_1e-3_F3_d32_U_O_c1_C_p1_S_p1_mads7i3a_up03"
ckpt = "02_s5_m160x160x160_1e-3_F3_d32_U_O_c1_C_p1_S_p1_mads7i3a_up03"
ckpt = "02_s5_m160x160x160_1e-3_F3_d32_U_O_c1_C_p1_s.1_r1_S_p1_mads7i3a_si27"
ckpt = "02_s5_m160x160x160_1e-4_F3_d32_U_V3r_n512_l1_O_c1_C_p1_s.1_r1_S_p1_mads7i3a_vq10"
ckpt = "02_s5_m160x160x160_1e-3_F3_d32_U_O_c1_C_p1_s.1_r1_S_p1_mads7i3a_si27"
ckpt = "02_s5_m160x160x160_1e-9_F3_d32_U_V3r_n512_l1_Of_c1_Cf_p1_s.1_r1_Sf_p1_mads7i3a_vq13"
ckpt = "02_s5_m160x160x160_1e-9_F3_d32_U_V3r_n512_l1_Of_c1_Cf_p1_s.1_r1_Sf_p1_mads7i3a_vq13"
ckpt = "04_s1_m160x160x160_1e-4_F3f_d32_U_V3rf_n512_G3v_c1_Of_Cf_Sf_mads7i3t_car13"
# ckpt = "02_s5_m160x160x160_1e-4_F3_d32_U_V3r_n512_l1_O_c1_C_p1_s.1_r1_S_p1_mads7i3a_vq10"
# ckpt = "01_s5_m160x160x160_1e-4_F3_d32_U_V3r_n512_l1_O_c1_C_p1_s.1_r1_S_p1_mads7i3a_vq14"
ckpt = "01_s5_m160x160x160_1e-4_F3_d32_U_V3r_n512_l1_O_c1_C_p1_s.1_r1_S_p1_mads7i3a_vq14"
ckpt = "01_s5_m160x160x160_1e-4_F3_d32_U_V3r_n512_l1_O_c1_C_p1_s.1_r1_S_p1_mads7i3a_vq14"
ckpt = "04_s1_m160x160x160_1e-4_F3f_d32_U_V3rf_n512_G3v_c1_Of_Cf_Sf_mads7i3t_car22"
# ckpt = "02_s5_m160x160x160_1e-4_F3_d32_U_V3r_n512_l1_O_c1_s.001_C_p1_s.1_r1_S_p1_s.001_mads7i3a_vq15"
ckpt = "01_s5_m160x160x160_1e-4_F3_d32_U_V3r_n512_l1_O_c1_C_p1_s.1_r1_S_p1_mads7i3a_vq14"
ckpt = "04_s1_m160x160x160_1e-4_F3f_d32_U_V3rf_n512_G3v_c1_Of_Cf_Sf_mads7i3t_car34"
ckpt = "01_s1_p128x384_1e-4_U_s.001_V_d64_e1_mafs7i3ep09_pr117"
ckpt = "01_s1_p128x384_1e-4_U_V_d64_e1_s.1_mafs7i3ep09_pr125"
ckpt = "02_s2_m160x80x160_1e-4_F3_d32_O_c1_E3_n2_c1_mags7i3t_taqs100i2v_com56"
ckpt = "02_s2_m128x64x128_1e-4_F3s_d32_O_c1_mags7i3t_pro38"
ckpt = "02_s2_m128x64x128_1e-3_F3s_d32_O_c1_s.1_mags7i3t_pro39"
ckpt = "02_s2_m128x64x128_1e-4_F3s_d32_O_c1_s.1_mags7i3t_pro41"
ckpt = "02_s2_m160x80x160_1e-4_F3_d32_O_c1_s.1_E3_n2_c.1_mags7i3t_pro43"
# ckpt = "02_s2_m160x80x160_1e-3_F3f_d32_Of_c1_M_p1_s.01_tars100i2t_rel61"
# ckpt = "02_s2_m128x16x128_1e-4_F3_d64_E3_n4_d16_c1_mabs7i3t_fags16i3v_moc15"
# ckpt = "02_s4_1e-4_P_c1_s.01_mags7i3t_occ09"
# ckpt = "02_s2_1e-4_P_c1_s.01_mags7i3t_occ09"
ckpt = "01_s2_m128x32x128_1e-4_F3_d32_G_2x11x2x1x2_r4_t1_d1_kafs8i1t_kafs64i1v_play02"
ckpt = "01_s2_m128x32x128_1e-4_F3_d32_G_2x11x2x1x2_r4_t1_d1_kafs8i1t_kafs64i1v_play21"
ckpt = "01_s2_m128x32x128_z32x32x32_1e-5_M_c1_r.1_F3_d32_kafs8i1ten_mat30"
ckpt = "01_s2_m128x32x128_1e-4_M_c1_r.1_F3_d8_kafs8i1t_kafs64i1v_mat39"
# ckpt = "01_s2_m128x32x128_1e-4_M_c1_r.1_F3_d8_kafs8i1t_kais100i1v_mat59"
ckpt = "01_s2_m128x32x128_1e-4_M_c1_r.1_F3_d8_kais8i1t_kais100i1v_mat64"
ckpt = "01_s2_m192x96x192_p128x384_1e-3_F3_d4_O_c1_R_d64_r10_mags7i3t_planA16"
ckpt = "01_s2_m192x96x192_p128x384_1e-3_F3_d4_O_c1_R_d64_r10_mags7i3t_planA17"
ckpt = "02_s2_m64x64x64_1e-4_S3i_c1_mags7i3t_unc03"
ckpt = "01_s2_m256x128x256_p128x384_1e-3_F3_d4_O_c1_R_d64_r10_mags7i3t_planA18"
ckpt = "01_s1_m192x96x192_p128x384_1e-3_F3_d4_O_c1_s.1_R_d64_r10_cacs10i2one_steal53"
ckpt = "01_s2_m128x64x128_1e-3_F3_d4_O_c1_s.1_cacs20i2one_bkg08"
ckpt = "01_s2_m128x64x128_1e-3_F3_d4_O_c1_s.1_cacs20i2one_feat01"
ckpt = "01_s2_m128x64x128_1e-3_F3_d4_O_c1_s.1_cacs20i2one_feat02"
ckpt = "01_s20_m128x64x128_1e-3_F3_d4_O_c1_t1_s.1_cacs20i2one_feat06"
# ckpt = "01_s20_m128x64x128_1e-3_F3_d4_O_c1_t10_s.1_cacs20i2one_feat07"
# ckpt = "01_s20_m128x64x128_1e-4_F3_d4_O_c1_t1_s.1_cacs20i2one_feat08"
# ckpt = "01_s2_m128x64x128_1e-5_F3_d4_O_c1_t1_s.1_cacs30i2one_bust09"
ckpt = "02_s2_m128x128x128_1e-4_F3_d64_O_c1_s.1_E3_n2_d16_c1_mags7i3t_bu12"
ckpt = "02_s2_m128x128x128_1e-4_F3_d64_O_c1_E3_n2_d16_c1_mags7i3t_bu21"
ckpt = "02_s2_m128x128x128_1e-4_F3_d64_O_c1_E3_n2_d16_c1_mags7i3t_bu20"
ckpt = "02_s2_m128x128x128_1e-4_F3_d64_O_c1_s.1_E3_n2_d16_c1_mags7i3t_bu26"
ckpt = "02_s2_m128x128x128_1e-4_F3_d64_O_c1_s.1_E3_n2_d16_c1_mags7i3t_bu28"
ckpt = "02_s2_m128x64x128_1e-4_F3_d64_O_c1_s.1_E3_n2_d16_c1_mags7i3t_bu31"
ckpt = "02_s2_m128x16x128_1e-4_F3_d64_E3_n4_d16_c1_mabs7i3t_fags16i3v_moc15"
ckpt = "02_s2_m128x128x128_1e-4_F3_d64_E3_n2_d16_c1_smabs5i8t_trainer_ml_2"
ckpt = "02_s2_m128x128x128_1e-4_L_F3_d64_E3_n2_d16_c1_smabs5i8t_trainer_implicit_2"



#baseline 2
ckpt = "02_s2_m128x128x128_1e-4_F3_d64_E3_n2_d16_c1_stahs50i1t_trainer_ml_occ_2"







#ml5
ckpt = '02_s2_m128x128x128_1e-3_L_F3_d64_E3_n2_d16_c1_smabs5i8t_trainer_implicit_lr3_2'

# mean_ious [0.98 0.39 0.31 0.27 0.27 0.23 0.2  0.18 0.16 0.3 ]
# all_mean_ious 0.3306653

#ml4
ckpt = '02_s2_m128x128x128_1e-4_L_F3_d64_O_c1_s.1_E3_n2_d16_c1_smabs5i8t_trainer_implicit_occ_1'

# mean_ious [0.98 0.72 0.59 0.57 0.53 0.53 0.44 0.55 0.49 0.43]
# all_mean_ious 0.58239716

# ml3
ckpt ='02_s2_m128x128x128_1e-4_L_F3_d64_E3_n2_d16_c1_smabs5i8t_trainer_implicit_2_again'
# mean_ious [0.99 0.82 0.77 0.68 0.6  0.72 0.73 0.59 0.63 0.58]
# all_mean_ious 0.71194315


#ml2
# ckpt ='02_s2_m128x128x128_1e-4_L_F3_d64_E3_n2_d16_c1_smabs5i8t_trainer_implicit_hyper_2'
# mean_ious [0.98 0.72 0.62 0.55 0.46 0.4  0.33 0.36 0.39 0.3 ]
# all_mean_ious 0.50987196

# #ml1
# ckpt ='02_s2_m128x128x128_1e-4_L_F3_d64_E3_n2_d16_c1_smabs5i8t_trainer_implicit_delta_2'
# # mean_ious [0.99 0.7  0.51 0.59 0.47 0.57 0.57 0.55 0.52 0.32]
# # all_mean_ious 0.5785452


# #random
# ckpt ='asdfjkasdhfkjasdf'
# # mean_ious [0.88 0.44 0.36 0.26 0.3  0.35 0.24 0.23 0.25 0.14]
# # all_mean_ious 0.343084


#baseline
ckpt ='02_s2_m128x128x128_1e-4_F3_d64_E3_n2_d16_c1_smabs5i8t_trainer_ml_occ_2'
# mean_ious [0.98 0.56 0.34 0.43 0.5  0.43 0.34 0.32 0.29 0.36]
# all_mean_ious 0.4571157


ckpt ='02_s2_m128x128x128_1e-4_L_F3_d64_E3_n2_d16_c1_complete_smallt_trainer_implicit_hyper_delta'
ckpt ='02_s2_m128x128x128_1e-4_L_F3_d64_E3_n2_d16_c1_complete_smallt_trainer_implicit_delta_nearest_concat'
ckpt ='02_s2_m128x128x128_1e-4_L_F3_d64_E3_n2_d16_c1_complete_smallt_trainer_implicit_delta_nearest'
ckpt ='02_s2_m128x128x128_1e-4_L_F3_d64_E3_n2_d16_c1_complete_smallt_trainer_implicit_hyper_delta_nearest'
ckpt ='02_s2_m128x128x128_1e-4_F3_d64_E3_n2_d16_c1_complete_smallt_trainer_ml'
# <<<<<<< HEAD

# =======
ckpt = '02_s2_m128x128x128_1e-4_L_F3_d64_E3_n2_d16_c1_smabs5i8t_trainer_implicit_2_again' #
ckpt ='02_s2_m128x128x128_1e-4_L_F3_d64_E3_n2_d16_c1_complete_smallt_trainer_implicit'
# >>>>>>> b7a69a7d3cb29a3e638303f401105af34f009c37
# #baseline
## ckpt ='02_s2_m128x128x128_1e-4_F3_d64_E3_n2_d16_c1_smabs5i8t_trainer_ml_2_again'
## mean_ious [0.86 0.68 0.65 0.64 0.68 0.64 0.61 0.48 0.4  0.44] 
## all_mean_ious 0.60909784
feat3d_init = ckpt
feat3d_dim = 32
ego_init = ckpt
localdecoder_init = ckpt
feat3docc_init = "02_s2_m128x128x128_5e-4_L_F3_d64_mc_cart_trainer_implicit_carla_5lr4_1"
localdecoder_render_init = "02_s2_m128x128x128_5e-4_L_F3_d64_mc_cart_trainer_implicit_carla_5lr4_1"
preocc_init = ckpt
feat2D_init = ckpt
feat3D_init = ckpt
up3D_init = ckpt
seg_init = ckpt
center_init = ckpt
vq3d_init = ckpt
sigen3d_init = ckpt
# gen3d_init = "04_s1_m160x160x160_1e-4_F3f_d32_U_V3rf_n512_G3v_c1_Of_Cf_Sf_mads7i3t_car13"
gen3d_init = ckpt
occ_init = ckpt
view_init = ckpt
loc_init = ckpt
match_init = ckpt
rigid_init = ckpt
motionreg_init = ckpt
mot_init = ckpt
occrel_init = "04_s2_m160x80x160_1e-3_F3f_d32_Of_c1_R_c1_mags7i3t_mags7i3v_rel35"
# conf_init = ckpt
conf_init = "04_s2_m256x128x256_z64x64x64_1e-4_F3f_d64_Mf_c1_r.1_C_c1_tals90i1t_tals90i1v_conf37"

feat2D_dim = 8
feat3D_dim = 32
view_depth = 64
vq3d_num_embeddings = 512

latents_init = "01_s1_p128x384_1e-3_O_c1_s.1_R_d64_c10_mafs7i3ep09_sa02"
latents_init = "01_s20_m64x32x64_p128x384_1e-2_O_c1_s.1_R_d64_r10_cacs20i2one_gv29"
latents_init = "01_s20_m128x64x128_1e-3_F3f_d4_O_c1_t1_s.1_cacs20i2one_traj12"
latents_init = "01_s20_m128x64x128_1e-2_F3f_d4_O_c1_t1_s.5_cacs20i2one_bust34"
latents_init = "01_s20_m128x64x128_p128x384_1e-2_O_c1_t1_s.5_R_d64_r10_cacs20i2one_bust54"
latents_init = "01_s20_m128x64x128_p128x384_1e-2_O_c1_t1_s.5_R_d64_r10_cacs20i2one_bust58"
latents_init = "01_s20_m128x64x128_p128x384_1e-2_O_c1_t1_s.5_R_d64_r10_cacs20i2one_bust58"
latents_init = "01_s20_m128x64x128_p128x384_1e-3_O_c1_t1_s.5_R_d64_r10_cacs20i2one_bust69"
latents_init = "01_s20_m128x64x128_p128x384_1e-3_L_oo2_bo1_os2_ss1_r1_O_c1_t1_s.5_R_d64_r1_cacs20i2one_bust80"

vq2d_init = "04_s1_m64x16x64_1e-3_Vr_n512_d64_r1_l1_L_c1_mabs7i3t_c21"
vq2d_num_embeddings = 512
# occ_init = "02_s2_m128x8x128_p64x192_1e-4_F64_V3r_n512_l1_O_c.1_s.01_V_d64_e1_mabs7i3t_mabs7i3v_r41"
# mean_ious [0.98 0.82 0.84 0.8  0.8  0.67 0.58 0.66 0.54 0.62]
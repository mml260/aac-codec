/* Stand-alone driver for GstPEAQ's core (basic version, mono, 48 kHz).
 * Mirrors process_fft_block_basic(), do_processing(), do_flush() and
 * calculate_odg() from gstpeaq.c line for line, minus the GStreamer plumbing.
 * Input: two raw little-endian float32 files (samples scaled to [-1, 1)). */
#include "fftearmodel.h"
#include "leveladapter.h"
#include "modpatt.h"
#include "movaccum.h"
#include "movs.h"
#include "nn.h"
#include <glib/gprintf.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

enum { MOVBASIC_BANDWIDTH_REF, MOVBASIC_BANDWIDTH_TEST, MOVBASIC_TOTAL_NMR,
  MOVBASIC_WIN_MOD_DIFF, MOVBASIC_ADB, MOVBASIC_EHS, MOVBASIC_AVG_MOD_DIFF_1,
  MOVBASIC_AVG_MOD_DIFF_2, MOVBASIC_RMS_NOISE_LOUD, MOVBASIC_MFPD,
  MOVBASIC_REL_DIST_FRAMES, COUNT_MOV_BASIC };
static const char *names[] = { "BandwidthRefB", "BandwidthTestB", "TotalNMRB",
  "WinModDiff1B", "ADBB", "EHSB", "AvgModDiff1B", "AvgModDiff2B",
  "RmsNoiseLoudB", "MFPDB", "RelDistFramesB" };

static PeaqEarModel *ear;
static gpointer ref_state[1], test_state[1];
static PeaqLevelAdapter *level[1];
static PeaqModulationProcessor *refmod[1], *testmod[1];
static PeaqMovAccum *acc[COUNT_MOV_BASIC];
static guint frame_counter = 0, loudness_reached_frame = G_MAXUINT;

static gboolean above_threshold (const gfloat *d, guint n)
{
  gfloat sum = 0; guint i;
  for (i = 0; i < 5; i++) sum += fabs (d[i]);
  while (i < n) {
    sum += fabs (d[i]) - fabs (d[i - 5]);
    if (sum >= 200. / 32768) return TRUE;
    i++;
  }
  return FALSE;
}

static void process_block (gfloat *r, gfloat *t)
{
  guint i, n = peaq_earmodel_get_frame_size (ear);
  gboolean above = above_threshold (r, n);
  for (i = 0; i < COUNT_MOV_BASIC; i++) peaq_movaccum_set_tentative (acc[i], !above);
  peaq_earmodel_process_block (ear, ref_state[0], r);
  peaq_earmodel_process_block (ear, test_state[0], t);
  peaq_leveladapter_process (level[0], peaq_earmodel_get_excitation (ear, ref_state[0]),
                             peaq_earmodel_get_excitation (ear, test_state[0]));
  peaq_modulationprocessor_process (refmod[0], peaq_earmodel_get_unsmeared_excitation (ear, ref_state[0]));
  peaq_modulationprocessor_process (testmod[0], peaq_earmodel_get_unsmeared_excitation (ear, test_state[0]));
  if (loudness_reached_frame == G_MAXUINT &&
      peaq_earmodel_calc_loudness (ear, ref_state[0]) > 0.1 &&
      peaq_earmodel_calc_loudness (ear, test_state[0]) > 0.1)
    loudness_reached_frame = frame_counter;
  if (frame_counter >= 24)
    peaq_mov_modulation_difference (refmod, testmod, acc[MOVBASIC_AVG_MOD_DIFF_1],
                                    acc[MOVBASIC_AVG_MOD_DIFF_2], acc[MOVBASIC_WIN_MOD_DIFF]);
  if (frame_counter >= 24 && frame_counter - 3 >= loudness_reached_frame)
    peaq_mov_noise_loudness (refmod, testmod, level, acc[MOVBASIC_RMS_NOISE_LOUD]);
  peaq_mov_bandwidth (ref_state, test_state, acc[MOVBASIC_BANDWIDTH_REF], acc[MOVBASIC_BANDWIDTH_TEST]);
  peaq_mov_nmr (PEAQ_FFTEARMODEL (ear), ref_state, test_state, acc[MOVBASIC_TOTAL_NMR],
                acc[MOVBASIC_REL_DIST_FRAMES]);
  peaq_mov_prob_detect (ear, ref_state, test_state, 1, acc[MOVBASIC_ADB], acc[MOVBASIC_MFPD]);
  peaq_mov_ehs (ear, ref_state, test_state, acc[MOVBASIC_EHS]);
  frame_counter++;
}

static gfloat *load (const char *path, long *n)
{
  FILE *f = fopen (path, "rb"); if (!f) { perror (path); exit (1); }
  fseek (f, 0, SEEK_END); *n = ftell (f) / sizeof (gfloat); fseek (f, 0, SEEK_SET);
  gfloat *d = g_new (gfloat, *n);
  if (fread (d, sizeof (gfloat), *n, f) != (size_t) *n) { perror ("read"); exit (1); }
  fclose (f); return d;
}

int main (int argc, char **argv)
{
  if (argc < 3) { fprintf (stderr, "usage: %s ref.f32 test.f32 [playback_level_dB]\n", argv[0]); return 2; }
  long nr, nt, pos = 0; guint i;
  gfloat *r = load (argv[1], &nr), *t = load (argv[2], &nt);
  double lvl = argc > 3 ? atof (argv[3]) : 92.0;
  ear = g_object_new (PEAQ_TYPE_FFTEARMODEL, NULL);
  g_object_set (ear, "playback-level", lvl, NULL);
  g_object_set (ear, "number-of-bands", 109, NULL);
  for (i = 0; i < COUNT_MOV_BASIC; i++) { acc[i] = peaq_movaccum_new (); peaq_movaccum_set_channels (acc[i], 1); }
  /* same accumulation modes as GstPeaq's basic-mode setup */
  peaq_movaccum_set_mode (acc[MOVBASIC_BANDWIDTH_REF], MODE_AVG);
  peaq_movaccum_set_mode (acc[MOVBASIC_BANDWIDTH_TEST], MODE_AVG);
  peaq_movaccum_set_mode (acc[MOVBASIC_TOTAL_NMR], MODE_AVG_LOG);
  peaq_movaccum_set_mode (acc[MOVBASIC_WIN_MOD_DIFF], MODE_AVG_WINDOW);
  peaq_movaccum_set_mode (acc[MOVBASIC_ADB], MODE_ADB);
  peaq_movaccum_set_mode (acc[MOVBASIC_EHS], MODE_AVG);
  peaq_movaccum_set_mode (acc[MOVBASIC_AVG_MOD_DIFF_1], MODE_AVG);
  peaq_movaccum_set_mode (acc[MOVBASIC_AVG_MOD_DIFF_2], MODE_AVG);
  peaq_movaccum_set_mode (acc[MOVBASIC_RMS_NOISE_LOUD], MODE_RMS);
  peaq_movaccum_set_mode (acc[MOVBASIC_MFPD], MODE_FILTERED_MAX);
  peaq_movaccum_set_mode (acc[MOVBASIC_REL_DIST_FRAMES], MODE_AVG);
  ref_state[0] = peaq_earmodel_state_alloc (ear);
  test_state[0] = peaq_earmodel_state_alloc (ear);
  level[0] = peaq_leveladapter_new (ear);
  refmod[0] = peaq_modulationprocessor_new (ear);
  testmod[0] = peaq_modulationprocessor_new (ear);
  long N = peaq_earmodel_get_frame_size (ear), S = peaq_earmodel_get_step_size (ear);
  long n = MIN (nr, nt);
  while (pos + N <= n) { process_block (r + pos, t + pos); pos += S; }
  if (pos < nr || pos < nt) {          /* do_flush: zero-pad the remainder */
    gfloat *pr = g_new0 (gfloat, N), *pt = g_new0 (gfloat, N);
    memcpy (pr, r + pos, sizeof (gfloat) * MIN (N, MAX (0, nr - pos)));
    memcpy (pt, t + pos, sizeof (gfloat) * MIN (N, MAX (0, nt - pos)));
    process_block (pr, pt);
  }
  gdouble movs[COUNT_MOV_BASIC];
  for (i = 0; i < COUNT_MOV_BASIC; i++) { movs[i] = peaq_movaccum_get_value (acc[i]); g_printf ("%s %f\n", names[i], movs[i]); }
  gdouble di = peaq_calculate_di_basic (movs);
  g_printf ("DI %f\nODG %f\n", di, peaq_calculate_odg (di));
  return 0;
}

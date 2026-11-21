#include "gst/fft/gstfftf64.h"
#include <math.h>
#include <stdlib.h>
GstFFTF64 *gst_fft_f64_new (gint len, gboolean inverse)
{ GstFFTF64 *s = g_new (GstFFTF64, 1); s->len = len; s->inverse = inverse; return s; }
void gst_fft_f64_free (GstFFTF64 *s) { g_free (s); }
/* in-place iterative radix-2 complex FFT; sign=-1 forward, +1 inverse */
static void cfft (double *re, double *im, int n, int sign)
{
  int i, j, k, m;
  for (i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { double t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (m = 2; m <= n; m <<= 1) {
    double ang = sign * 2.0 * M_PI / m;
    for (k = 0; k < n; k += m)
      for (j = 0; j < m / 2; j++) {
        double wr = cos (ang * j), wi = sin (ang * j);
        double xr = re[k + j + m / 2], xi = im[k + j + m / 2];
        double tr = xr * wr - xi * wi, ti = xr * wi + xi * wr;
        re[k + j + m / 2] = re[k + j] - tr; im[k + j + m / 2] = im[k + j] - ti;
        re[k + j] += tr; im[k + j] += ti;
      }
  }
}
void gst_fft_f64_fft (GstFFTF64 *s, const gdouble *t, GstFFTF64Complex *f)
{
  int n = s->len, k;
  g_assert ((n & (n - 1)) == 0);
  double *re = g_new (double, n), *im = g_new0 (double, n);
  for (k = 0; k < n; k++) re[k] = t[k];
  cfft (re, im, n, -1);
  for (k = 0; k <= n / 2; k++) { f[k].r = re[k]; f[k].i = im[k]; }
  g_free (re); g_free (im);
}
void gst_fft_f64_inverse_fft (GstFFTF64 *s, const GstFFTF64Complex *f, gdouble *t)
{
  int n = s->len, k;
  double *re = g_new (double, n), *im = g_new (double, n);
  for (k = 0; k <= n / 2; k++) { re[k] = f[k].r; im[k] = f[k].i; }
  for (k = n / 2 + 1; k < n; k++) { re[k] = f[n - k].r; im[k] = -f[n - k].i; }
  cfft (re, im, n, +1);
  for (k = 0; k < n; k++) t[k] = re[k];
  g_free (re); g_free (im);
}

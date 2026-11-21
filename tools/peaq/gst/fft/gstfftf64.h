/* Minimal stand-in for GStreamer's GstFFTF64 API (radix-2 FFT), so GstPEAQ's
 * core can be built without GStreamer. Same conventions as gst_fft_f64:
 * forward is unnormalized with e^{-j}, real input -> len/2+1 bins;
 * inverse is unnormalized, len/2+1 bins -> real output. */
#ifndef SHIM_GSTFFTF64_H
#define SHIM_GSTFFTF64_H
#include <glib.h>
typedef struct { gdouble r; gdouble i; } GstFFTF64Complex;
typedef struct { gint len; gboolean inverse; } GstFFTF64;
GstFFTF64 *gst_fft_f64_new (gint len, gboolean inverse);
void gst_fft_f64_fft (GstFFTF64 *self, const gdouble *timedata, GstFFTF64Complex *freqdata);
void gst_fft_f64_inverse_fft (GstFFTF64 *self, const GstFFTF64Complex *freqdata, gdouble *timedata);
void gst_fft_f64_free (GstFFTF64 *self);
#endif

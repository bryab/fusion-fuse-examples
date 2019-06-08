/*
----------------------------------------------------------------------
Copyright (c) 2012, Stefan Ihringer
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met: 

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
----------------------------------------------------------------------

This file must match the fuse's file name, just with a .cl extension instead of .fuse
*/


/*** Linear Gradient between 2 Points */
__kernel
void cl_lineargradient(FuWriteImage_t dst,
	const int2 imgsize,				// size of full image
	const int4 outwnd,				// .xy = output window offset, .zw = size
	const float4 points,			// .xy = 1st point, .zw = 2nd point relative to 1st point
	const float4 color1,
	const float4 color2,
	const float2 factors,			// .x = inverted distance squared, .y = scaling factor that accounts for Fusion's coordinate system
	const int extrapolate)			// 1 = extrapolate gradient beyond two points
{
	// current position (integer pixels, starting with (0/0) at the bottom left corner of the image's DoD
	const int2 ipos = (int2)(get_global_id(1), get_global_id(0));
	// current position as a vector relative to starting point of gradient
	const float2 p = convert_float2(ipos + outwnd.xy) / convert_float2(imgsize) * (float2)(1.0, factors.y) - points.s01;
	// position along gradient [0..1] = length of current position vector projected onto gradient vector
	float t = dot(p, points.s23) * factors.x;
	if (extrapolate == 0)
	{
		t = clamp(t, 0.0f, 1.0f);
	}
	FuWriteImagef(dst, ipos, outwnd.zw, color1 * (1 - t) + color2 * t);
}


/*** Barycentric Gradient between 3 Points */
__kernel
void cl_barygradient(FuWriteImage_t dst,
	const int2 imgsize,				// size of full image
	const int4 outwnd,				// .xy = output window offset, .zw = size
	const float4 pointsA,			// .s01 = 1st point, .s23 = 2nd point (relative to 1st point)
	const float2 pointsB,			// .s01 = 3rd point (relative to 1st point)
	const float4 color1,
	const float4 color2,
	const float4 color3,
	const float2 factors,			// .x = inverted pre-calculated area of triangle, .y = scaling factor that accounts for Fusion's coordinate system
	const int extrapolate)			// 1 = extrapolate gradient beyond two points
{
	// current position (integer pixels, starting with (0/0) at the bottom left corner of the image's DoD
	const int2 ipos = (int2)(get_global_id(1), get_global_id(0));
	// current position as a vector relative to starting point of gradient
	const float2 p = convert_float2(ipos + outwnd.xy) / convert_float2(imgsize) * (float2)(1.0, factors.y) - pointsA.s01;
	// area of sub-triangles opposing each corner
	float a3 = cross((float4)(pointsA.s2, pointsA.s3, 0, 0), (float4)(p.x, p.y, 0, 0)).z * factors.x;
	float a2 = cross((float4)(p.x, p.y, 0, 0), (float4)(pointsB.s0, pointsB.s1, 0, 0)).z * factors.x;
	float a1 = 1.0 - a2 - a3;
	if (extrapolate == 0)
	{
		// distribute contribution of a negative sub-triangle to the other two colors
		if (a1 < 0)
		{
			a2 = a2 + a1 * 0.5;
			a3 = a3 + a1 * 0.5;
		}
		if (a2 < 0)
		{
			a1 = a1 + a2 * 0.5;
			a3 = a3 + a2 * 0.5;
		}
		if (a3 < 0)
		{
			a1 = a1 + a3 * 0.5;
			a2 = a2 + a3 * 0.5;
		}
		a1 = clamp(a1, 0.0f, 1.0f);
		a2 = clamp(a2, 0.0f, 1.0f);
		a3 = clamp(a3, 0.0f, 1.0f);
	}
	FuWriteImagef(dst, ipos, outwnd.zw, color1 * a1 + color2 * a2 + color3 * a3);
}


// multiplies vector by matrix
float4 MatMultVec(const float4 vec, const float16 mat)
{
	float4 out;
	out.x = dot(vec, mat.lo.lo);
	out.y = dot(vec, mat.lo.hi);
	out.z = dot(vec, mat.hi.lo);
	out.w = dot(vec, mat.hi.hi);
	return out;
}


/*** Perspective 4-Corner Gradient (like a corner pin) */
__kernel
void cl_fourcornergradient(FuWriteImage_t dst,
	const int2 imgsize,				// size of full image
	const int4 outwnd,				// .xy = output window offset, .zw = size
	const float16 mat,				// inverted projection matrix
	const float4 color1,
	const float4 color2,
	const float4 color3,
	const float4 color4,
	const int extrapolate)			// 1 = extrapolate gradient beyond two points
{
	// current position (integer pixels, starting with (0/0) at the bottom left corner of the image's DoD
	const int2 ipos = (int2)(get_global_id(1), get_global_id(0));
	// current position as a vector
	float4 p = (float4)(convert_float(ipos.x + outwnd.x) / convert_float(imgsize.x), convert_float(ipos.y + outwnd.y) / convert_float(imgsize.y), 1.0, 1.0);
	// p becomes position inside unit square, where four colors are located at (0/0), (1/0), (1/1) and (0/1)
	p = MatMultVec(p, mat);
	p.z = fmax(0.001f, fabs(p.z));
	p.xy /= p.zz;
	if (extrapolate == 0)
	{
		p.xy = clamp(p.xy, 0.0f, 1.0f);
	}
	// color5 & color6: colors of points along top and bottom edge
	const float4 color5 = color1 * (1 - p.x) + color2 * p.x;
	const float4 color6 = color4 * (1 - p.x) + color3 * p.x;
	FuWriteImagef(dst, ipos, outwnd.zw, color6 * (1 - p.y) + color5 * p.y);
}


/*** Inverse Distance Gradient (Shepard's method) */
__kernel
void cl_inversedistancegradient(FuWriteImage_t dst, __global float2* points, __global float4* colors,
	const int2 imgsize,				// size of full image
	const int4 outwnd,				// .xy = output window offset, .zw = size
	const int count,				// number of points/colors
	const float2 factors)			// .x = power factor for distance weighting, .y = y-aspect correction factor
{
	// current position (integer pixels, starting with (0/0) at the bottom left corner of the image's DoD
	const int2 ipos = (int2)(get_global_id(1), get_global_id(0));
	// current position as a vector
	const float2 p = convert_float2(ipos + outwnd.xy) / convert_float2(imgsize) * (float2)(1.0, factors.y);
	float4 result = (float4)(0,0,0,0);
	float summed_weights = 0;
	for (int i = 0; i < count; i++)
	{
		const float d = length(p - points[i]);
		if (fabs(d) < 0.00001)
		{
			result = colors[i];
			summed_weights = 1;
			i = count;
		}
		else
		{
			const float weight = 1.0 / powr(d, factors.x);
			summed_weights += weight;
			result += colors[i] * weight;
		}
	}
	FuWriteImagef(dst, ipos, outwnd.zw, result / summed_weights);
}


/*** Voronoi Gradient (color of nearest seed point) */
__kernel
void cl_voronoigradient(FuWriteImage_t dst, __global float2* points, __global float4* colors,
	const int2 imgsize,				// size of full image
	const int4 outwnd,				// .xy = output window offset, .zw = size
	const int count,				// number of points/colors
	const int distancemap,			// 1 = output distance map in alpha channel
	const float aspect)				// y-aspect correction factor
{
	// current position (integer pixels, starting with (0/0) at the bottom left corner of the image's DoD
	const int2 ipos = (int2)(get_global_id(1), get_global_id(0));
	// current position as a vector
	const float2 p = convert_float2(ipos + outwnd.xy) / convert_float2(imgsize) * (float2)(1.0, aspect);
	float4 result;
	float smallest_d = MAXFLOAT;
	for (int i = 0; i < count; i++)
	{
		const float d = length(p - points[i]);
		if( isless(d, smallest_d) )
		{
			smallest_d = d;
			result = colors[i];
		}
	}
	if (distancemap == 1)
	{
		result.w = smallest_d;
	}
	FuWriteImagef(dst, ipos, outwnd.zw, result);
}

#!/usr/bin/python

#/*************************************************************************/
#/*                                                                       */
#/*                  Language Technologies Institute                      */
#/*                     Carnegie Mellon University                        */
#/*                        Copyright (c) 2006                             */
#/*                        All Rights Reserved.                           */
#/*                                                                       */
#/*  Permission is hereby granted, free of charge, to use and distribute  */
#/*  this software and its documentation without restriction, including   */
#/*  without limitation the rights to use, copy, modify, merge, publish,  */
#/*  distribute, sublicense, and/or sell copies of this work, and to      */
#/*  permit persons to whom this work is furnished to do so, subject to   */
#/*  the following conditions:                                            */
#/*   1. The code must retain the above copyright notice, this list of    */
#/*      conditions and the following disclaimer.                         */
#/*   2. Any modifications must be clearly marked as such.                */
#/*   3. Original authors' names are not deleted.                         */
#/*   4. The authors' names are not used to endorse or promote products   */
#/*      derived from this software without specific prior written        */
#/*      permission.                                                      */
#/*                                                                       */
#/*  CARNEGIE MELLON UNIVERSITY AND THE CONTRIBUTORS TO THIS WORK         */
#/*  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      */
#/*  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   */
#/*  SHALL CARNEGIE MELLON UNIVERSITY NOR THE CONTRIBUTORS BE LIABLE      */
#/*  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    */
#/*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   */
#/*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          */
#/*  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       */
#/*  THIS SOFTWARE.                                                       */
##/*                                                                      */
#/*************************************************************************/
#/*             Author:                                                   */
#/*               Date:  Noveber 2016                                     */
#/*************************************************************************/


import wave,struct,numpy,numpy.fft
import scipy.signal
import pyaudio
import os

test_flag = 1
tts_debug_flag = 1

class speech_io:

  def __init__(self):
    self.sig = []	

  def wavread(self, wav_name, start_time=0, end_time=0):
      '''
      Input       : path to the wave file
      start_time  : Start time in seconds
      end_time    : End time in seconds
        
      '''
      wave_filepointer = wave.open(wav_name,'r')
      fs = wave_filepointer.getframerate() 
      sig = []
    
      # Get frames
      wav_frames = wave_filepointer.getnframes()
    
      # Convert times to frames
      start_frame = int(start_time * fs)
      end_frame = int(end_time * fs)

      wave_filepointer.setpos(start_frame)

      # Boundary Condition
      if end_frame == 0:
    	 end_frame = wav_frames

      # Read the data
      for i in range(start_frame, end_frame):
         sig.append(float(struct.unpack('h',wave_filepointer.readframes(1))[0]))

      wave_filepointer.close()
    
      return numpy.array(sig,dtype='float'),fs 


  def wavplay(self, wav_name, start_time=0, end_time=0):
       '''
       Input       : path to the wave file
       start_time  : Start time in seconds
       end_time    : End time in seconds
       '''
       #wave_file = self.wavread(wav_name, start_time, end_time)
   
       wave_filepointer = wave.open(wav_name,'r')

       # Parameters
       chunk = 1024

       #instantiate PyAudio  
       p = pyaudio.PyAudio()  
       
       #open stream  
       stream = p.open(format = p.get_format_from_width(wave_filepointer.getsampwidth()),  
                channels = wave_filepointer.getnchannels(),  
                rate = wave_filepointer.getframerate(),  
                output = True)  

       # Read data  
       data = wave_filepointer.readframes(chunk)  


       # Play stream  
       while data != '':  
           stream.write(data)  
           data = wave_filepointer.readframes(chunk)  

       # Stop stream  
       stream.stop_stream()  
       stream.close()  

       # Close PyAudio  
       p.terminate()  



class MCEP:
    

    def __init__(self, number_filterbanks, lower_cutoff=None, higher_cutoff=None, N=512, fs = 16000):
        '''
        Input Parameters:
        number_filterbanks     : Number of filter banks
        N                      : N point DFT
        lower_cutoff           : Lower cutoff frequency
        higher_cutoff          : Higher cutoff frequency
        fs                     : Sampling Frequency

        Note: Following are the standard cutoff frequencies:
        1. fs = 8000 Hz, LOFREQ = 300, HIFREQ = 3400 #given in HTK manual
        2. fs = 16000 Hz, LOFREQ = 133.33334, HIFREQ=6855.4976 #default parameters in sphinx3
        '''

        self.fs = float(fs)
        self.number_filterbanks = number_filterbanks
        self.N = N
        if lower_cutoff == None:
            self.lower_cutoff = 0

        if lower_cutoff > self.fs/2 or lower_cutoff < 0:
            print 'lower_cutoff is not in permissable range'
            print 'Resetting lower_cutoff to 0'
            lower_cutoff = 0

        self.lower_cutoff = lower_cutoff

        if higher_cutoff == None:
            self.higher_cutoff = self.fs/2

        if higher_cutoff > self.fs/2 or higher_cutoff < 0:
            print 'higher_cutoff is not in permissable range'
            print 'Resetting HIFREQ to nyquist frequency'
            higher_cutoff = self.fs/2

        self.higher_cutoff = higher_cutoff

        if self.lower_cutoff >= self.higher_cutoff:
            print 'Bad frequency ranges given'
            sys.exit(1)
            
        self.__melfilterbanks()

    def __mel(self,linf):
        return (2595.0 * numpy.log10(1.0 + (linf/700.0)))

    def __melinv(self,melf):
        return (700.0 * (numpy.power(10.0,melf/2595.0) - 1))

    def __melfilterbanks(self):
        lower_mel = self.__mel(self.lower_cutoff)
        higher_mel = self.__mel(self.higher_cutoff)

        melwidth = (higher_mel - lower_mel)/(self.number_filterbanks + 1)

        mel_filters = []
        for i in range(0,self.number_filterbanks + 2):                  
            mel_filters.append(lower_mel + (i * melwidth))
        mel_filters = numpy.array(mel_filters)

        linear_filters = self.__melinv(mel_filters) 
        filter_banks = (linear_filters * self.N)/self.fs 

        self.LON = filter_banks[0]   
        self.HIN = filter_banks[-1]  
        
        self.H = numpy.array([[0.0] * self.N for i in xrange(self.number_filterbanks + 2)]) 

        for i in range(1,self.number_filterbanks + 1):
            for k in range(0,self.N):
                if k <= filter_banks[i-1]:
                    self.H[i][k] = 0
                elif filter_banks[i-1] <= k <= filter_banks[i]:
                    self.H[i][k] = (k - filter_banks[i-1])/(filter_banks[i] - filter_banks[i-1])
                elif filter_banks[i] <= k <= filter_banks[i+1]:
                    self.H[i][k] = (filter_banks[i+1] - k)/(filter_banks[i+1] - filter_banks[i])
                else:
                    self.H[i][k] = 0


    def melspec(self,sig):
        '''
        Computes the Mel Spectrum values for the windowed speech signal.
        Computation of the Mel Spectrum is as follows:
        
                               N - 1
                       X[i] = SUMMATION |S(k)| * H[i][k]
                                 k=0

        where H is weighting function.
        Input Parameters:
        sig : windowed speech signal
        '''
        X = numpy.zeros(self.number_filterbanks+1)
        for i in range(1,len(X)):
            X[i] = sum(numpy.abs(numpy.fft.fft(sig,self.N)) * self.H[i])
        return X[1:]

   
    def melspecfeats(self,sig,window_length,window_shift):
      
        window_length_frames = (window_length * self.fs)/1000
        window_shift_frames = (window_shift * self.fs)/1000
        
        sig = numpy.append(sig[0],numpy.diff(sig)) #to remove dc
        sig = sig + 0.001 
        noFrames = int((len(sig) - window_length_frames)/window_shift_frames) + 1
        
        mspec = []

        for i in range(0,noFrames):
            index = int(i * window_shift_frames)
            window_signal = sig[int(index):int(index+window_length_frames)]
            smooth_signal = window_signal * numpy.hamming(len(window_signal))
            mspec.append(self.melspec(smooth_signal))
        mspec = numpy.array(mspec)
        return mspec

 


   



if test_flag == 1:

        if tts_debug_flag ==1:
	   os.system('flite -t "Okay, I am testing this script"' )

	# Instantiate the class
	s = speech_io()
	sig = s.wavread('../crazy.wav')

	# Play back
        if tts_debug_flag ==1:
     	   os.system('flite -t "Let me try to read a wavefile and play it back"' )
	s.wavplay('../crazy.wav')

        if tts_debug_flag ==1:
       	    os.system('flite -t "That seems fine, although I must have produced a voicing error. Ok Let me extract mel cepstrum"' )

	mcep = MCEP(40)
	window_length = 10
	window_shift = 5
	print mcep.melspecfeats(sig,window_length,window_shift)

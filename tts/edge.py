import time
import asyncio
import numpy as np
import resampy
import soundfile as sf
import edge_tts
import subprocess
from threading import Thread
from io import BytesIO

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

@register("tts", "edgetts")
class EdgeTTS(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]):
        text,textevent = msg
        voice = self.opt.REF_FILE or "zh-CN-YunxiaNeural" 
        voicename = textevent.get('tts', {}).get('ref_file',voice) #self.opt.REF_FILE #"zh-CN-YunxiaNeural"
        if getattr(self.parent, 'is_streaming_avatar', False):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    self.__stream_to_ditto(voicename, text, textevent)
                )
            finally:
                loop.close()
            return

        t = time.time()
        asyncio.new_event_loop().run_until_complete(self.__main(voicename,text))
        logger.info(f'-------edge tts time:{time.time()-t:.4f}s')
        if self.input_stream.getbuffer().nbytes<=0: #edgetts err
            logger.error('edgetts err!!!!!')
            return
        
        self.input_stream.seek(0)
        stream = self.__create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk and self.state==State.RUNNING:
            eventpoint={}
            streamlen -= self.chunk
            if idx==0:
                eventpoint={'status':'start','text':text}
            elif streamlen<self.chunk:
                eventpoint={'status':'end','text':text}
            eventpoint.update(**textevent) #eventpoint={'status':'end','text':text,'msgevent':textevent}
            self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
            idx += self.chunk
        #if streamlen>0:  #skip last frame(not 20ms)
        #    self.queue.put(stream[idx:])
        self.input_stream.seek(0)
        self.input_stream.truncate() 

    async def __stream_to_ditto(self, voicename, text, textevent):
        command = [
            'ffmpeg', '-loglevel', 'error',
            '-f', 'mp3', '-i', 'pipe:0',
            '-f', 's16le', '-acodec', 'pcm_s16le',
            '-ar', str(self.sample_rate), '-ac', '1', 'pipe:1',
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        reader_errors = []
        emitted_audio = [False]

        def enqueue_pcm(block, first=False, final=False):
            if self.state != State.RUNNING:
                return
            eventpoint = {'text': text}
            if first:
                eventpoint['status'] = 'start'
            if final:
                eventpoint['status'] = 'end'
            eventpoint.update(**textevent)
            pcm = np.frombuffer(block, dtype='<i2').astype(np.float32) / 32768.0
            self.parent.put_audio_frame(pcm, eventpoint)
            emitted_audio[0] = True

        def read_pcm():
            pending = None
            first = True
            remainder = bytearray()
            try:
                while True:
                    data = process.stdout.read1(4096)
                    if not data:
                        break
                    remainder.extend(data)
                    while len(remainder) >= self.chunk * 2:
                        block = bytes(remainder[:self.chunk * 2])
                        del remainder[:self.chunk * 2]
                        if pending is not None:
                            enqueue_pcm(pending, first=first)
                            first = False
                        pending = block
                if remainder:
                    remainder.extend(bytes(self.chunk * 2 - len(remainder)))
                    block = bytes(remainder)
                    if pending is not None:
                        enqueue_pcm(pending, first=first)
                        first = False
                    pending = block
                if pending is not None:
                    enqueue_pcm(pending, first=first, final=True)
            except Exception as error:
                reader_errors.append(error)

        reader = Thread(target=read_pcm, daemon=True, name="edge-tts-pcm-reader")
        reader.start()
        try:
            communicate = edge_tts.Communicate(text, voicename)
            async for chunk in communicate.stream():
                if self.state != State.RUNNING:
                    break
                if chunk.get('type') == 'audio':
                    process.stdin.write(chunk['data'])
                    process.stdin.flush()
        except Exception:
            process.terminate()
            process.wait()
            reader.join()
            logger.exception('edge tts streaming to Ditto failed')
            raise
        finally:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()

        reader.join()
        stderr = process.stderr.read()
        return_code = process.wait()
        if reader_errors:
            raise RuntimeError('Could not stream decoded Edge TTS audio') from reader_errors[0]
        if return_code:
            raise RuntimeError(
                f"FFmpeg could not decode Edge TTS audio: "
                f"{stderr.decode('utf-8', errors='replace').strip()}"
            )
        if not emitted_audio[0] and self.state == State.RUNNING:
            raise RuntimeError('Edge TTS returned no decodable audio for Ditto')

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream
    
    async def __main(self,voicename: str, text: str):
        try:
            communicate = edge_tts.Communicate(text, voicename)

            #with open(OUTPUT_FILE, "wb") as file:
            first = True
            async for chunk in communicate.stream():
                if first:
                    first = False
                if chunk["type"] == "audio" and self.state==State.RUNNING:
                    #self.push_audio(chunk["data"])
                    self.input_stream.write(chunk["data"])
                    #file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
        except Exception as e:
            logger.exception('edgetts')

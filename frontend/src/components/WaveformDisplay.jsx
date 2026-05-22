import { useEffect, useRef } from 'react'

export default function WaveformDisplay({ isRecording, analyserNode }) {
  const canvasRef = useRef(null)
  const animFrameRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    if (!isRecording || !analyserNode) {
      cancelAnimationFrame(animFrameRef.current)
      ctx.fillStyle = '#f0f0f0'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      return
    }

    const dataArray = new Uint8Array(analyserNode.fftSize)

    const draw = () => {
      animFrameRef.current = requestAnimationFrame(draw)
      analyserNode.getByteTimeDomainData(dataArray)

      ctx.fillStyle = '#f0f0f0'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      ctx.lineWidth = 2
      ctx.strokeStyle = '#000000'
      ctx.beginPath()

      const sliceWidth = canvas.width / dataArray.length
      let x = 0

      for (let i = 0; i < dataArray.length; i++) {
        const v = dataArray[i] / 128.0
        const y = (v * canvas.height) / 2
        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
        x += sliceWidth
      }

      ctx.lineTo(canvas.width, canvas.height / 2)
      ctx.stroke()
    }

    draw()

    return () => cancelAnimationFrame(animFrameRef.current)
  }, [isRecording, analyserNode])

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={96}
      className="w-full h-24 rounded-lg"
    />
  )
}

export async function compareVoice(audioBlob) {
  const formData = new FormData()
  formData.append("audio", audioBlob, "recording.webm")
  const response = await fetch("/api/compare", {
    method: "POST",
    body: formData,
  })
  const data = await response.json()
  if (!response.ok) {
    throw data
  }
  return data
}

export async function getHealth() {
  const response = await fetch("/api/health")
  return response.json()
}

export async function getSpectrogram(vocalistId) {
  const response = await fetch(`/api/spectrogram/${vocalistId}`)
  return response.json()
}

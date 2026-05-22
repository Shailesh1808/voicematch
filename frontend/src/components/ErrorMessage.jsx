export default function ErrorMessage({ message }) {
  return (
    <div className="bg-red-50 border border-red-100 rounded-lg p-4">
      <p className="text-sm text-red-600">{message}</p>
    </div>
  )
}

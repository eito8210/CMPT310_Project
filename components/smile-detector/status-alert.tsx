import { Alert, AlertDescription, AlertTitle } from "../ui/alert"
import { Terminal } from "lucide-react"

type StatusAlertProps = {
  modelError: string | null
  modelsLoaded: boolean
}

export function StatusAlert({ modelError, modelsLoaded }: StatusAlertProps) {
  if (modelError) {
    return (
      <Alert variant="destructive">
        <Terminal className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{modelError}</AlertDescription>
      </Alert>
    )
  }

  if (!modelsLoaded) {
    return (
      <Alert>
        <Terminal className="h-4 w-4" />
        <AlertTitle>Initializing</AlertTitle>
        <AlertDescription>Loading face recognition models. Please wait...</AlertDescription>
      </Alert>
    )
  }

  return null
}

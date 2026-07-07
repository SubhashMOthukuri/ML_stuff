import { Component } from 'react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  render() {
    if (this.state.error) {
      return (
        <div className="min-h-screen bg-[#080810] flex items-center justify-center p-8">
          <div className="max-w-md w-full rounded-2xl border border-red-500/25 bg-red-500/8 p-8 text-center">
            <p className="text-red-300 font-bold text-lg mb-2">Something went wrong</p>
            <p className="text-red-400/70 text-sm font-mono break-all">
              {this.state.error?.message ?? String(this.state.error)}
            </p>
            <button
              onClick={() => this.setState({ error: null })}
              className="mt-6 px-4 py-2 rounded-xl border border-red-500/30 bg-red-500/15
                         text-red-300 text-sm hover:bg-red-500/25 transition-colors">
              Try again
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

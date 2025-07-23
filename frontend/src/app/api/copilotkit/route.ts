import {
  CopilotRuntime,
  copilotRuntimeNextJSAppRouterEndpoint,
  OpenAIAdapter,
  GoogleGenerativeAIAdapter
} from '@copilotkit/runtime';
import { NextRequest } from 'next/server';
import { HttpAgent } from "@ag-ui/client";


const llamaIndexAgent = new HttpAgent({
  url: process.env.NEXT_PUBLIC_LLAMAINDEX_URL || "http://0.0.0.0:8000/llamaindex-agent",
  // url: "https://open-ag-ui-demo-llamaindex.onrender.com/llamaindex-agent",
});
const serviceAdapter = new GoogleGenerativeAIAdapter()
const runtime = new CopilotRuntime({
  agents: {
    // @ts-ignore
    llamaIndexAgent : llamaIndexAgent 
  },
});
// const runtime = new CopilotRuntime()
export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: '/api/copilotkit',
  });

  return handleRequest(req);
};
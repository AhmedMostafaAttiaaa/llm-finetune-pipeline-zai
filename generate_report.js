const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, HeadingLevel, BorderStyle, ShadingType,
  PageBreak, Header, Footer, PageNumber, NumberFormat, TableOfContents,
  ImageRun, TabStopPosition, TabStopType, convertInchesToTwip,
  SectionType
} = require("docx");
const fs = require("fs");

// ================================================================
// PALETTE: Dawn Mist Tech (cool, professional, tech)
// ================================================================
const palette = {
  primary: "1B3A4B",      // Deep teal-navy
  body: "2C3E50",         // Dark charcoal
  secondary: "5D6D7E",    // Steel gray
  accent: "2980B9",       // Bright blue
  surface: "EBF5FB",      // Light blue-white
  white: "FFFFFF",
  lightGray: "F2F3F4",
  medGray: "D5D8DC",
  darkText: "1C2833",
  tableHeader: "1B3A4B",
  tableAlt: "EBF5FB",
  warning: "E67E22",
  success: "27AE60",
  danger: "C0392B",
};

// ================================================================
// FONTS
// ================================================================
const fonts = {
  heading: "Calibri",
  body: "Calibri",
  chineseBody: "Microsoft YaHei",
  mono: "Consolas",
};

// ================================================================
// HELPER: Safe text
// ================================================================
function safeText(v) {
  if (v === null || v === undefined) return "";
  return String(v);
}

// ================================================================
// COVER PAGE (R4 - Top Color Block)
// ================================================================
function buildCover() {
  const topBlockHeight = 6000;
  return [
    // Top color block
    new Table({
      rows: [
        new TableRow({
          height: { value: topBlockHeight, rule: "exact" },
          children: [
            new TableCell({
              width: { size: 100, type: WidthType.PERCENTAGE },
              shading: { type: ShadingType.CLEAR, fill: palette.primary },
              borders: {
                top: { style: BorderStyle.NONE, size: 0 },
                bottom: { style: BorderStyle.NONE, size: 0 },
                left: { style: BorderStyle.NONE, size: 0 },
                right: { style: BorderStyle.NONE, size: 0 },
              },
              children: [
                new Paragraph({ spacing: { before: 1200 } }),
                new Paragraph({
                  alignment: AlignmentType.CENTER,
                  spacing: { after: 200 },
                  children: [
                    new TextRun({
                      text: "LLM Fine-Tuning Pipeline",
                      font: fonts.heading,
                      size: 52,
                      bold: true,
                      color: palette.white,
                    }),
                  ],
                }),
                new Paragraph({
                  alignment: AlignmentType.CENTER,
                  spacing: { after: 400 },
                  children: [
                    new TextRun({
                      text: "Model Selection & Evaluation Report",
                      font: fonts.heading,
                      size: 36,
                      color: "AED6F1",
                    }),
                  ],
                }),
                new Paragraph({
                  alignment: AlignmentType.CENTER,
                  spacing: { after: 100 },
                  children: [
                    new TextRun({
                      text: "Coding & Fintech Domain",
                      font: fonts.body,
                      size: 28,
                      color: "D4E6F1",
                    }),
                  ],
                }),
                new Paragraph({
                  alignment: AlignmentType.CENTER,
                  spacing: { after: 100 },
                  children: [
                    new TextRun({
                      text: "LoRA/QLoRA with Unsloth | Single GPU (30GB VRAM)",
                      font: fonts.body,
                      size: 22,
                      color: "D4E6F1",
                    }),
                  ],
                }),
              ],
            }),
          ],
        }),
      ],
      width: { size: 100, type: WidthType.PERCENTAGE },
    }),
    // Bottom section with metadata
    new Paragraph({ spacing: { before: 600 } }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 100 },
      children: [
        new TextRun({
          text: "Prepared: May 2026",
          font: fonts.body,
          size: 22,
          color: palette.secondary,
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 100 },
      children: [
        new TextRun({
          text: "8 Candidate Models | 7 Benchmarks | Full VRAM Analysis",
          font: fonts.body,
          size: 20,
          color: palette.secondary,
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [
        new TextRun({
          text: "Chain-of-Thought | Self-Reflection | Tool Use | Multi-Step Reasoning",
          font: fonts.mono,
          size: 18,
          color: palette.accent,
        }),
      ],
    }),
  ];
}

// ================================================================
// HEADING HELPERS
// ================================================================
function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 400, after: 200 },
    children: [
      new TextRun({
        text: safeText(text),
        font: fonts.heading,
        size: 32,
        bold: true,
        color: palette.primary,
      }),
    ],
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 300, after: 150 },
    children: [
      new TextRun({
        text: safeText(text),
        font: fonts.heading,
        size: 26,
        bold: true,
        color: palette.accent,
      }),
    ],
  });
}

function heading3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 100 },
    children: [
      new TextRun({
        text: safeText(text),
        font: fonts.heading,
        size: 22,
        bold: true,
        color: palette.body,
      }),
    ],
  });
}

function bodyText(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 312 },
    alignment: AlignmentType.JUSTIFIED,
    children: [
      new TextRun({
        text: safeText(text),
        font: fonts.body,
        size: 21,
        color: opts.color || palette.body,
        bold: opts.bold || false,
        italics: opts.italic || false,
      }),
    ],
  });
}

function monoText(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 100, line: 276 },
    indent: { left: 400 },
    children: [
      new TextRun({
        text: safeText(text),
        font: fonts.mono,
        size: 18,
        color: opts.color || palette.body,
      }),
    ],
  });
}

function bulletPoint(text, level = 0) {
  return new Paragraph({
    spacing: { after: 80, line: 312 },
    indent: { left: 600 + level * 300, hanging: 300 },
    children: [
      new TextRun({ text: "\u2022 ", font: fonts.body, size: 21, color: palette.accent }),
      new TextRun({ text: safeText(text), font: fonts.body, size: 21, color: palette.body }),
    ],
  });
}

// ================================================================
// TABLE HELPERS
// ================================================================
function makeHeaderCell(text, width) {
  return new TableCell({
    width: { size: width, type: WidthType.PERCENTAGE },
    shading: { type: ShadingType.CLEAR, fill: palette.tableHeader },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 1, color: palette.primary },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: palette.primary },
      left: { style: BorderStyle.SINGLE, size: 1, color: palette.medGray },
      right: { style: BorderStyle.SINGLE, size: 1, color: palette.medGray },
    },
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({
            text: safeText(text),
            font: fonts.heading,
            size: 18,
            bold: true,
            color: palette.white,
          }),
        ],
      }),
    ],
  });
}

function makeDataCell(text, width, opts = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.PERCENTAGE },
    shading: opts.shaded ? { type: ShadingType.CLEAR, fill: palette.tableAlt } : undefined,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 1, color: palette.medGray },
      bottom: { style: BorderStyle.SINGLE, size: 1, color: palette.medGray },
      left: { style: BorderStyle.SINGLE, size: 1, color: palette.medGray },
      right: { style: BorderStyle.SINGLE, size: 1, color: palette.medGray },
    },
    children: [
      new Paragraph({
        alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
        children: [
          new TextRun({
            text: safeText(text),
            font: opts.mono ? fonts.mono : fonts.body,
            size: opts.mono ? 16 : 18,
            color: opts.color || palette.body,
            bold: opts.bold || false,
          }),
        ],
      }),
    ],
  });
}

// ================================================================
// MODEL DATA
// ================================================================
const models = [
  {
    name: "Qwen2.5-Coder-1.5B-Instruct",
    short: "qwen2.5-coder-1.5b",
    size: "1.5B",
    family: "Qwen2",
    context: "32K",
    baseVram: "~3 GB",
    trainVram: "6-8 GB",
    fit30GB: "Yes (comfortable)",
    reasoning: "Basic",
    coding: 9,
    fintech: 7,
    strengths: "Excellent code generation for its size, fast iteration, low VRAM, ideal for prototyping",
    weaknesses: "Limited complex reasoning, shorter context than larger models, may struggle with nuanced fintech analysis",
    bestFor: "Rapid prototyping, edge deployment, fast experimentation cycles, limited VRAM environments",
  },
  {
    name: "Phi-2 (2.7B)",
    short: "phi-2",
    size: "2.7B",
    family: "Phi",
    context: "2K",
    baseVram: "~5.5 GB",
    trainVram: "8-10 GB",
    fit30GB: "Yes (comfortable)",
    reasoning: "Moderate",
    coding: 7,
    fintech: 6,
    strengths: "Strong reasoning for size, good at math and logic, Microsoft Research quality",
    weaknesses: "Very short context (2048), base model needs instruction tuning first, limited code breadth",
    bestFor: "Reasoning-heavy tasks, math problems, quick experiments with small models",
  },
  {
    name: "Qwen2.5-Coder-7B-Instruct",
    short: "qwen2.5-coder-7b",
    size: "7B",
    family: "Qwen2",
    context: "128K",
    baseVram: "~14 GB",
    trainVram: "16-18 GB",
    fit30GB: "Yes (good margin)",
    reasoning: "Strong",
    coding: 10,
    fintech: 9,
    strengths: "Best-in-class 7B coding model, massive 128K context, beats many larger models on code benchmarks, excellent instruction following",
    weaknesses: "Moderate VRAM required, less specialized reasoning than DeepSeek-R1",
    bestFor: "Coding + fintech production deployment, balanced performance, long-context code analysis",
  },
  {
    name: "Llama-3.1-8B-Instruct",
    short: "llama-3.1-8b",
    size: "8B",
    family: "Llama3",
    context: "128K",
    baseVram: "~16 GB",
    trainVram: "18-20 GB",
    fit30GB: "Yes (QLoRA only)",
    reasoning: "Strong",
    coding: 8,
    fintech: 9,
    strengths: "Strong all-around model, excellent reasoning and instruction following, huge community and ecosystem, 128K context",
    weaknesses: "QLoRA 4-bit required for 30GB VRAM, less coding-specialized than Qwen-Coder, LoRA FP16 is too tight",
    bestFor: "General fintech applications, conversational AI, well-rounded tasks with strong reasoning",
  },
  {
    name: "DeepSeek-Coder-V2-Lite (7B MoE)",
    short: "deepseek-coder-7b",
    size: "7B MoE",
    family: "DeepSeek",
    context: "128K",
    baseVram: "~14 GB",
    trainVram: "16-18 GB",
    fit30GB: "Yes (good margin)",
    reasoning: "Strong",
    coding: 9,
    fintech: 8,
    strengths: "MoE architecture (efficient inference), excellent code performance, strong analytical reasoning, long context",
    weaknesses: "MoE complexity, requires trust_remote_code, LoRA target modules differ from standard",
    bestFor: "Coding-heavy tasks, efficient inference at scale, fintech code generation with lower compute cost",
  },
  {
    name: "DeepSeek-R1-Distill-Qwen-7B",
    short: "deepseek-r1-qwen-7b",
    size: "7B",
    family: "DeepSeek-R1",
    context: "128K",
    baseVram: "~14 GB",
    trainVram: "16-20 GB",
    fit30GB: "Yes (good margin)",
    reasoning: "Exceptional",
    coding: 8,
    fintech: 9,
    strengths: "Built-in chain-of-thought reasoning, natural self-reflection, multi-step problem solving, outputs <think/> blocks natively, distilled from R1 reasoning model",
    weaknesses: "Verbose outputs (thinking tokens), can overthink simple questions, slightly higher VRAM usage during training",
    bestFor: "REASONING & THINKING (primary goal), complex fintech analysis, self-correction, multi-step problem solving",
  },
  {
    name: "Mistral-7B-Instruct-v0.3",
    short: "mistral-7b",
    size: "7B",
    family: "Mistral",
    context: "32K",
    baseVram: "~14 GB",
    trainVram: "16-18 GB",
    fit30GB: "Yes (good margin)",
    reasoning: "Strong",
    coding: 8,
    fintech: 8,
    strengths: "Battle-tested reliability, excellent efficiency and speed, sliding window attention, great for production, strong community support",
    weaknesses: "Less specialized than coding-specific models, sliding window may lose very long context",
    bestFor: "Production systems requiring reliability, fast inference, balanced coding + fintech workloads",
  },
  {
    name: "Qwen2.5-Coder-14B-Instruct",
    short: "qwen2.5-coder-14b",
    size: "14B",
    family: "Qwen2",
    context: "128K",
    baseVram: "~28 GB",
    trainVram: "24-28 GB",
    fit30GB: "Tight (QLoRA 4-bit only, batch=1)",
    reasoning: "Very Strong",
    coding: 10,
    fintech: 9,
    strengths: "Highest code quality, largest model that fits 30GB with QLoRA, best performance on code benchmarks, 128K context",
    weaknesses: "TIGHT on 30GB VRAM, must use QLoRA 4-bit with batch_size=1, slowest training, reduced LoRA rank (32 vs 64), reduced seq_length (2048 vs 4096)",
    bestFor: "Maximum code quality when VRAM allows, complex coding tasks requiring deep understanding",
  },
];

// ================================================================
// BUILD DOCUMENT
// ================================================================
async function buildDocument() {
  const doc = new Document({
    styles: {
      default: {
        document: {
          run: { font: fonts.body, size: 21, color: palette.body },
          paragraph: { spacing: { line: 312 } },
        },
        heading1: {
          run: { font: fonts.heading, size: 32, bold: true, color: palette.primary },
          paragraph: { spacing: { before: 400, after: 200 } },
        },
        heading2: {
          run: { font: fonts.heading, size: 26, bold: true, color: palette.accent },
          paragraph: { spacing: { before: 300, after: 150 } },
        },
        heading3: {
          run: { font: fonts.heading, size: 22, bold: true, color: palette.body },
          paragraph: { spacing: { before: 200, after: 100 } },
        },
      },
    },
    sections: [
      // ============================================================
      // SECTION 1: COVER PAGE
      // ============================================================
      {
        properties: {
          page: {
            margin: { top: 0, bottom: 0, left: 0, right: 0 },
          },
        },
        children: buildCover(),
      },

      // ============================================================
      // SECTION 2: TOC
      // ============================================================
      {
        properties: {
          page: {
            margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
          },
        },
        headers: {
          default: new Header({
            children: [
              new Paragraph({
                alignment: AlignmentType.RIGHT,
                children: [
                  new TextRun({
                    text: "LLM Fine-Tuning Pipeline | Model Selection Report",
                    font: fonts.body,
                    size: 16,
                    color: palette.secondary,
                    italics: true,
                  }),
                ],
              }),
            ],
          }),
        },
        footers: {
          default: new Footer({
            children: [
              new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [
                  new TextRun({ text: "Page ", font: fonts.body, size: 16, color: palette.secondary }),
                  new TextRun({ children: [PageNumber.CURRENT], font: fonts.body, size: 16, color: palette.secondary }),
                ],
              }),
            ],
          }),
        },
        children: [
          new Paragraph({
            spacing: { after: 300 },
            children: [
              new TextRun({
                text: "Table of Contents",
                font: fonts.heading,
                size: 32,
                bold: true,
                color: palette.primary,
              }),
            ],
          }),
          new TableOfContents("Table of Contents", {
            hyperlink: true,
            headingStyleRange: "1-3",
          }),
          new Paragraph({
            spacing: { before: 200 },
            children: [
              new TextRun({
                text: "Hint: Right-click the table of contents and select \u201cUpdate Field\u201d to refresh page numbers.",
                font: fonts.body,
                size: 16,
                italics: true,
                color: palette.secondary,
              }),
            ],
          }),
          new Paragraph({
            children: [new PageBreak()],
          }),
        ],
      },

      // ============================================================
      // SECTION 3: BODY
      // ============================================================
      {
        properties: {
          page: {
            margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
            pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL },
          },
        },
        headers: {
          default: new Header({
            children: [
              new Paragraph({
                alignment: AlignmentType.RIGHT,
                children: [
                  new TextRun({
                    text: "LLM Fine-Tuning Pipeline | Model Selection Report",
                    font: fonts.body,
                    size: 16,
                    color: palette.secondary,
                    italics: true,
                  }),
                ],
              }),
            ],
          }),
        },
        footers: {
          default: new Footer({
            children: [
              new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [
                  new TextRun({ text: "Page ", font: fonts.body, size: 16, color: palette.secondary }),
                  new TextRun({ children: [PageNumber.CURRENT], font: fonts.body, size: 16, color: palette.secondary }),
                ],
              }),
            ],
          }),
        },
        children: [
          // ========================================================
          // 1. EXECUTIVE SUMMARY
          // ========================================================
          heading1("1. Executive Summary"),

          bodyText(
            "This report presents a comprehensive analysis of eight open-source large language models (LLMs) suitable for fine-tuning in the coding and fintech domain. All models are evaluated against the constraint of a single GPU with 30GB VRAM, using LoRA and QLoRA fine-tuning methods accelerated by the Unsloth framework. The pipeline covers the complete lifecycle from dataset preparation through fine-tuning to multi-benchmark evaluation, with explicit support for chain-of-thought reasoning, self-reflection, tool use, and multi-step problem solving."
          ),

          bodyText(
            "The evaluation framework encompasses seven benchmarks: MMLU (general knowledge including finance), GSM8K (mathematical reasoning), HumanEval (code generation), MBPP (Python programming), MT-Bench (multi-turn conversation with GPT-as-judge), FinQA (financial question answering), and ConvFinQA (conversational financial reasoning). Additionally, perplexity scoring and custom fintech evaluation categories (regulatory compliance, risk assessment, algorithmic trading, fraud detection, and portfolio optimization) are included."
          ),

          bodyText(
            "Our top recommendation for the primary goal of fine-tuning reasoning and thinking is DeepSeek-R1-Distill-Qwen-7B, which offers built-in chain-of-thought reasoning and natural self-reflection capabilities. For the best balance of coding performance and reasoning, Qwen2.5-Coder-7B-Instruct is the optimal choice. For maximum code quality within the VRAM budget, Qwen2.5-Coder-14B-Instruct can be used with QLoRA 4-bit and careful memory management."
          ),

          // Key findings box
          new Paragraph({
            spacing: { before: 200, after: 100 },
            shading: { type: ShadingType.CLEAR, fill: palette.surface },
            indent: { left: 200, right: 200 },
            children: [
              new TextRun({
                text: "KEY FINDINGS",
                font: fonts.heading,
                size: 20,
                bold: true,
                color: palette.primary,
              }),
            ],
            border: {
              left: { style: BorderStyle.SINGLE, size: 12, color: palette.accent, space: 10 },
            },
          }),

          bulletPoint("Best for Reasoning: DeepSeek-R1-Distill-Qwen-7B \u2014 built-in <think/> blocks, chain-of-thought, and self-reflection"),
          bulletPoint("Best for Coding: Qwen2.5-Coder-7B-Instruct \u2014 top code benchmarks in 7B class, 128K context"),
          bulletPoint("Best Overall Balance: Qwen2.5-Coder-7B-Instruct \u2014 strong coding + solid reasoning + comfortable VRAM margin"),
          bulletPoint("Maximum Quality: Qwen2.5-Coder-14B-Instruct \u2014 highest code quality, but tight on 30GB VRAM (QLoRA 4-bit only, batch=1)"),
          bulletPoint("Fastest Iteration: Qwen2.5-Coder-1.5B-Instruct \u2014 ideal for prototyping and rapid experimentation"),
          bulletPoint("Embedding Models: BAAI/bge-large-en-v1.5 recommended for fintech semantic search and retrieval augmentation"),

          // ========================================================
          // 2. INTRODUCTION & SCOPE
          // ========================================================
          heading1("2. Introduction and Scope"),

          heading2("2.1 Project Overview"),
          bodyText(
            "The goal of this project is to build a complete, reproducible pipeline for fine-tuning and evaluating open-source large language models from Hugging Face, specifically targeting the coding and fintech business domain. The pipeline must support fine-tuning of reasoning and thinking capabilities, including chain-of-thought (CoT) reasoning, self-reflection and self-correction, tool use and function calling, and complex multi-step problem solving. Additionally, the pipeline includes support for NLP embedding models used in retrieval-augmented generation (RAG) and semantic search workflows."
          ),

          heading2("2.2 Hardware Constraints"),
          bodyText(
            "The entire pipeline is designed to operate within the constraints of a single GPU with a maximum of 30GB VRAM. This constraint significantly influences model selection, training strategy, and hyperparameter choices. Specifically, it means that 7B-8B models require QLoRA 4-bit quantization for training, the 14B model requires QLoRA 4-bit with batch_size=1 and reduced sequence length, and LoRA FP16 training is only feasible for models under 3B parameters. The Unsloth framework is used to maximize training efficiency and reduce memory overhead through optimized attention kernels, gradient checkpointing, and 2x faster inference."
          ),

          heading2("2.3 Fine-Tuning Approach"),
          bodyText(
            "The pipeline employs two primary fine-tuning strategies. The first is QLoRA 4-bit (Quantized Low-Rank Adaptation), which loads the base model in 4-bit NormalFloat4 (NF4) quantization with double quantization for weight savings, then applies LoRA adapters in BF16 precision. This approach reduces VRAM usage by approximately 70% compared to full fine-tuning while maintaining 97-99% of model quality. The second strategy is LoRA FP16, which is only applicable to the smallest models (1-3B) and provides slightly higher quality at the cost of significantly more VRAM. Both strategies use the Unsloth framework for 2x faster training and inference, optimized gradient checkpointing, and automatic mixed precision handling."
          ),

          bodyText(
            "For reasoning and thinking fine-tuning, the pipeline implements several augmentation strategies. First, when a thinking/reasoning column is available in the dataset, it is wrapped in <think/>...</think/> tags following the DeepSeek-R1 format, teaching the model to produce explicit reasoning traces. Second, when no reasoning column exists, chain-of-thought scaffolding is added to the system prompt, encouraging step-by-step problem decomposition. Third, self-reflection prompts can be injected to teach the model to verify and correct its own outputs. These strategies can be combined and are configurable per training run."
          ),

          // ========================================================
          // 3. MODEL COMPARISON
          // ========================================================
          heading1("3. Model Comparison and Analysis"),

          heading2("3.1 Overview Table"),
          bodyText(
            "The following table provides a high-level comparison of all eight candidate models, including key specifications, VRAM requirements, and suitability ratings for coding, fintech, and reasoning tasks. All VRAM estimates assume QLoRA 4-bit training with Unsloth optimizations on a single GPU."
          ),

          // Main comparison table
          new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            rows: [
              new TableRow({
                tableHeader: true,
                children: [
                  makeHeaderCell("Model", 18),
                  makeHeaderCell("Size", 7),
                  makeHeaderCell("Context", 7),
                  makeHeaderCell("Train VRAM", 10),
                  makeHeaderCell("Fit 30GB?", 9),
                  makeHeaderCell("Coding", 7),
                  makeHeaderCell("Fintech", 7),
                  makeHeaderCell("Reasoning", 9),
                  makeHeaderCell("Chat Template", 10),
                  makeHeaderCell("Type", 8),
                  makeHeaderCell("Trust RC", 8),
                ],
              }),
              ...models.map((m, i) =>
                new TableRow({
                  children: [
                    makeDataCell(m.short, 18, { mono: true, bold: true, shaded: i % 2 === 1 }),
                    makeDataCell(m.size, 7, { center: true, shaded: i % 2 === 1 }),
                    makeDataCell(m.context, 7, { center: true, shaded: i % 2 === 1 }),
                    makeDataCell(m.trainVram, 10, { center: true, shaded: i % 2 === 1 }),
                    makeDataCell(m.fit30GB, 9, { center: true, shaded: i % 2 === 1, color: m.fit30GB.includes("Tight") ? palette.warning : palette.success }),
                    makeDataCell(m.coding + "/10", 7, { center: true, shaded: i % 2 === 1 }),
                    makeDataCell(m.fintech + "/10", 7, { center: true, shaded: i % 2 === 1 }),
                    makeDataCell(m.reasoning, 9, { center: true, shaded: i % 2 === 1, color: m.reasoning === "Exceptional" ? palette.accent : m.reasoning === "Very Strong" ? palette.success : palette.body }),
                    makeDataCell(m.family === "Qwen2" ? "ChatML" : m.family === "Llama3" ? "Llama3" : m.family === "DeepSeek-R1" ? "DeepSeek" : "Custom", 10, { center: true, shaded: i % 2 === 1 }),
                    makeDataCell(m.family === "Phi" ? "Base" : "Instruct", 8, { center: true, shaded: i % 2 === 1 }),
                    makeDataCell(m.family === "Phi" || m.family === "DeepSeek" || m.family === "DeepSeek-R1" ? "Yes" : "No", 8, { center: true, shaded: i % 2 === 1 }),
                  ],
                })
              ),
            ],
          }),

          // ========================================================
          // 3.2 Detailed Model Profiles
          // ========================================================
          heading2("3.2 Detailed Model Profiles"),

          ...models.flatMap((m) => [
            heading3(m.name),
            bodyText(
              `The ${m.name} is a ${m.size} parameter model from the ${m.family} family, supporting a context length of ${m.context} tokens. ` +
              `It requires approximately ${m.trainVram} of VRAM for QLoRA 4-bit training with Unsloth, making it ` +
              (m.fit30GB.includes("Tight") ? "a tight fit on a 30GB GPU requiring careful memory management" : "comfortable to train on a 30GB GPU") +
              `.`
            ),
            bodyText(
              `Strengths: ${m.strengths}. Weaknesses: ${m.weaknesses}. This model is best suited for: ${m.bestFor}.`
            ),
          ]),

          // ========================================================
          // 4. VRAM ANALYSIS
          // ========================================================
          heading1("4. VRAM Budget Analysis"),

          heading2("4.1 Memory Breakdown"),
          bodyText(
            "Understanding the VRAM budget is critical for successful fine-tuning on a 30GB GPU. The total VRAM consumption during training consists of several components: the model weights (which depend on quantization), the LoRA adapter parameters (typically 0.5-2% of total parameters), optimizer states (AdamW 8-bit uses 8 bytes per trainable parameter), gradient storage, and activation memory (reduced by gradient checkpointing). The following table provides a detailed breakdown for each model."
          ),

          new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            rows: [
              new TableRow({
                tableHeader: true,
                children: [
                  makeHeaderCell("Model", 16),
                  makeHeaderCell("Base Model (4-bit)", 12),
                  makeHeaderCell("LoRA Adapter", 10),
                  makeHeaderCell("Optimizer (8-bit)", 12),
                  makeHeaderCell("Gradients", 10),
                  makeHeaderCell("Activations (GC)", 12),
                  makeHeaderCell("Total Estimated", 12),
                  makeHeaderCell("Headroom", 8),
                  makeHeaderCell("Risk Level", 8),
                ],
              }),
              // Qwen2.5-Coder-1.5B
              new TableRow({
                children: [
                  makeDataCell("qwen2.5-coder-1.5b", 16, { mono: true }),
                  makeDataCell("~1.2 GB", 12, { center: true }),
                  makeDataCell("~0.1 GB", 10, { center: true }),
                  makeDataCell("~0.3 GB", 12, { center: true }),
                  makeDataCell("~0.2 GB", 10, { center: true }),
                  makeDataCell("~3-4 GB", 12, { center: true }),
                  makeDataCell("6-8 GB", 12, { center: true }),
                  makeDataCell("22+ GB", 8, { center: true, color: palette.success }),
                  makeDataCell("None", 8, { center: true, color: palette.success }),
                ],
              }),
              // Phi-2
              new TableRow({
                children: [
                  makeDataCell("phi-2", 16, { mono: true, shaded: true }),
                  makeDataCell("~2.0 GB", 12, { center: true, shaded: true }),
                  makeDataCell("~0.15 GB", 10, { center: true, shaded: true }),
                  makeDataCell("~0.4 GB", 12, { center: true, shaded: true }),
                  makeDataCell("~0.3 GB", 10, { center: true, shaded: true }),
                  makeDataCell("~4-5 GB", 12, { center: true, shaded: true }),
                  makeDataCell("8-10 GB", 12, { center: true, shaded: true }),
                  makeDataCell("20+ GB", 8, { center: true, color: palette.success, shaded: true }),
                  makeDataCell("None", 8, { center: true, color: palette.success, shaded: true }),
                ],
              }),
              // Qwen2.5-Coder-7B
              new TableRow({
                children: [
                  makeDataCell("qwen2.5-coder-7b", 16, { mono: true }),
                  makeDataCell("~4.5 GB", 12, { center: true }),
                  makeDataCell("~0.3 GB", 10, { center: true }),
                  makeDataCell("~0.8 GB", 12, { center: true }),
                  makeDataCell("~0.5 GB", 10, { center: true }),
                  makeDataCell("~8-10 GB", 12, { center: true }),
                  makeDataCell("16-18 GB", 12, { center: true }),
                  makeDataCell("12+ GB", 8, { center: true, color: palette.success }),
                  makeDataCell("Low", 8, { center: true, color: palette.success }),
                ],
              }),
              // Llama-3.1-8B
              new TableRow({
                children: [
                  makeDataCell("llama-3.1-8b", 16, { mono: true, shaded: true }),
                  makeDataCell("~5.0 GB", 12, { center: true, shaded: true }),
                  makeDataCell("~0.3 GB", 10, { center: true, shaded: true }),
                  makeDataCell("~0.8 GB", 12, { center: true, shaded: true }),
                  makeDataCell("~0.5 GB", 10, { center: true, shaded: true }),
                  makeDataCell("~10-12 GB", 12, { center: true, shaded: true }),
                  makeDataCell("18-20 GB", 12, { center: true, shaded: true }),
                  makeDataCell("10+ GB", 8, { center: true, color: palette.success, shaded: true }),
                  makeDataCell("Low", 8, { center: true, color: palette.success, shaded: true }),
                ],
              }),
              // DeepSeek-Coder-V2-Lite
              new TableRow({
                children: [
                  makeDataCell("deepseek-coder-7b", 16, { mono: true }),
                  makeDataCell("~4.5 GB", 12, { center: true }),
                  makeDataCell("~0.3 GB", 10, { center: true }),
                  makeDataCell("~0.8 GB", 12, { center: true }),
                  makeDataCell("~0.5 GB", 10, { center: true }),
                  makeDataCell("~8-10 GB", 12, { center: true }),
                  makeDataCell("16-18 GB", 12, { center: true }),
                  makeDataCell("12+ GB", 8, { center: true, color: palette.success }),
                  makeDataCell("Low", 8, { center: true, color: palette.success }),
                ],
              }),
              // DeepSeek-R1
              new TableRow({
                children: [
                  makeDataCell("deepseek-r1-qwen-7b", 16, { mono: true, shaded: true }),
                  makeDataCell("~4.5 GB", 12, { center: true, shaded: true }),
                  makeDataCell("~0.3 GB", 10, { center: true, shaded: true }),
                  makeDataCell("~0.8 GB", 12, { center: true, shaded: true }),
                  makeDataCell("~0.5 GB", 10, { center: true, shaded: true }),
                  makeDataCell("~8-12 GB", 12, { center: true, shaded: true }),
                  makeDataCell("16-20 GB", 12, { center: true, shaded: true }),
                  makeDataCell("10+ GB", 8, { center: true, color: palette.success, shaded: true }),
                  makeDataCell("Low", 8, { center: true, color: palette.success, shaded: true }),
                ],
              }),
              // Mistral-7B
              new TableRow({
                children: [
                  makeDataCell("mistral-7b", 16, { mono: true }),
                  makeDataCell("~4.5 GB", 12, { center: true }),
                  makeDataCell("~0.3 GB", 10, { center: true }),
                  makeDataCell("~0.8 GB", 12, { center: true }),
                  makeDataCell("~0.5 GB", 10, { center: true }),
                  makeDataCell("~8-10 GB", 12, { center: true }),
                  makeDataCell("16-18 GB", 12, { center: true }),
                  makeDataCell("12+ GB", 8, { center: true, color: palette.success }),
                  makeDataCell("Low", 8, { center: true, color: palette.success }),
                ],
              }),
              // Qwen2.5-Coder-14B
              new TableRow({
                children: [
                  makeDataCell("qwen2.5-coder-14b", 16, { mono: true, shaded: true }),
                  makeDataCell("~8.5 GB", 12, { center: true, shaded: true }),
                  makeDataCell("~0.2 GB", 10, { center: true, shaded: true }),
                  makeDataCell("~0.6 GB", 12, { center: true, shaded: true }),
                  makeDataCell("~0.4 GB", 10, { center: true, shaded: true }),
                  makeDataCell("~12-16 GB", 12, { center: true, shaded: true }),
                  makeDataCell("24-28 GB", 12, { center: true, shaded: true }),
                  makeDataCell("2-6 GB", 8, { center: true, color: palette.warning, shaded: true }),
                  makeDataCell("HIGH", 8, { center: true, color: palette.danger, shaded: true }),
                ],
              }),
            ],
          }),

          heading2("4.2 Risk Assessment"),
          bodyText(
            "The VRAM risk assessment categorizes each model into three tiers. The 'None' risk tier includes models under 3B parameters, which have ample VRAM headroom for larger batch sizes, longer sequences, and even LoRA FP16 training. The 'Low' risk tier covers 7B-8B models with QLoRA 4-bit, which maintain a comfortable 10+ GB headroom on a 30GB GPU, allowing for reasonable batch sizes (2-4) and sequence lengths (2048-4096). The 'HIGH' risk tier applies only to the 14B model, which has just 2-6 GB of headroom, requiring batch_size=1, reduced LoRA rank (32 instead of 64), and max_seq_length of 2048 to avoid out-of-memory errors. Training the 14B model should be monitored carefully using the included GPU monitoring utility."
          ),

          bodyText(
            "For all models, we recommend enabling gradient checkpointing (Unsloth's optimized version), using AdamW 8-bit optimizer, and running the GPU monitor script in the background during training. The pipeline automatically configures these settings based on the selected model and training configuration. If an out-of-memory error occurs during 14B training, reducing the sequence length to 1024 or the LoRA rank to 16 can provide additional memory savings."
          ),

          // ========================================================
          // 5. DATASET PREPARATION
          // ========================================================
          heading1("5. Dataset Preparation Guide"),

          heading2("5.1 CSV Column Requirements"),
          bodyText(
            "The pipeline's data preparation script automatically detects columns in your CSV file by matching column names against known aliases. The minimum required columns are an instruction/user column and a response/assistant column. The following table shows all recognized column names and their purposes."
          ),

          new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            rows: [
              new TableRow({
                tableHeader: true,
                children: [
                  makeHeaderCell("Role", 15),
                  makeHeaderCell("Required?", 10),
                  makeHeaderCell("Recognized Column Names", 40),
                  makeHeaderCell("Description", 35),
                ],
              }),
              ...[
                ["instruction", "Required", "instruction, user, human, question, prompt, input, query, ask", "The user's question or prompt to the model"],
                ["response", "Required", "response, assistant, answer, output, reply, completion, target, solution", "The expected model response or answer"],
                ["system", "Optional", "system, context, background, preprompt, system_prompt", "System message setting model behavior and persona"],
                ["thinking", "Optional", "thinking, reasoning, explanation, chain, chain_of_thought, rationale, thought, cot", "Chain-of-thought reasoning trace for the response"],
                ["tool_calls", "Optional", "tool_calls, tools, function_calls, tool_use, functions", "Tool or function call specifications"],
                ["category", "Optional", "category, domain, type, topic, label, class", "Data category for stratified evaluation"],
              ].map((row, i) =>
                new TableRow({
                  children: [
                    makeDataCell(row[0], 15, { bold: true, mono: true, shaded: i % 2 === 1 }),
                    makeDataCell(row[1], 10, { center: true, shaded: i % 2 === 1, color: row[1] === "Required" ? palette.danger : palette.success }),
                    makeDataCell(row[2], 40, { shaded: i % 2 === 1 }),
                    makeDataCell(row[3], 35, { shaded: i % 2 === 1 }),
                  ],
                })
              ),
            ],
          }),

          heading2("5.2 Step-by-Step Preparation Process"),
          bodyText("Follow these steps to prepare your CSV dataset for fine-tuning:"),
          bulletPoint("Step 1: Validate your CSV using the data validator script. Run: python data/data_validator.py --input your_data.csv"),
          bulletPoint("Step 2: Review the validation report. Address any issues (empty values, duplicates, too short/long entries)."),
          bulletPoint("Step 3: If your column names are not auto-detected, specify them explicitly. Run: python data/prepare_dataset.py --input your_data.csv --columns instruction=YourCol response=YourCol"),
          bulletPoint("Step 4: The script will generate data in all formats (ChatML, Alpaca, ShareGPT, Llama3, DeepSeek) with train/val/test splits."),
          bulletPoint("Step 5: For reasoning fine-tuning, ensure you have a 'thinking' or 'chain_of_thought' column. If not present, the script will add CoT scaffolding automatically."),
          bulletPoint("Step 6: Verify the output by checking the preparation_stats.json file and reviewing sample entries."),

          heading2("5.3 Data Quality Recommendations"),
          bodyText(
            "For optimal fine-tuning results in the coding and fintech domain, we recommend a minimum of 500 samples for basic LoRA fine-tuning, with 2,000+ samples for strong results and 5,000+ samples for production-quality models. The dataset should maintain a balanced distribution across coding and fintech categories. For coding tasks, include a mix of code generation, code explanation, debugging, and refactoring examples. For fintech tasks, include regulatory compliance, risk assessment, financial analysis, algorithmic trading, and fraud detection examples. Each sample should include a detailed thinking/reasoning trace where possible, as this is the most important factor for reasoning fine-tuning quality."
          ),

          bodyText(
            "If your dataset is small (under 500 samples), consider augmenting it with synthetic data generated by a larger model (e.g., GPT-4 or Claude), using self-instruct or Evol-Instruct techniques. The pipeline's data preparation script includes automatic chain-of-thought augmentation for samples without explicit reasoning traces, and self-reflection prompt injection for improving self-correction capabilities."
          ),

          // ========================================================
          // 6. EVALUATION FRAMEWORK
          // ========================================================
          heading1("6. Evaluation Framework"),

          heading2("6.1 Benchmark Suite"),
          bodyText(
            "The evaluation pipeline implements a comprehensive benchmark suite covering general knowledge, mathematical reasoning, code generation, conversational quality, and domain-specific financial tasks. Each benchmark is designed to measure different aspects of model performance relevant to the coding and fintech domain."
          ),

          new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            rows: [
              new TableRow({
                tableHeader: true,
                children: [
                  makeHeaderCell("Benchmark", 12),
                  makeHeaderCell("Category", 12),
                  makeHeaderCell("Description", 30),
                  makeHeaderCell("Key Metric", 12),
                  makeHeaderCell("Relevance to Coding+Fintech", 22),
                  makeHeaderCell("Few-shot", 6),
                  makeHeaderCell("Tool", 6),
                ],
              }),
              ...[
                ["MMLU", "General Knowledge", "57 subjects including finance, accounting, econometrics, computer science", "acc_norm", "Tests broad knowledge including finance and CS domains", "5", "lm-eval"],
                ["GSM8K", "Math Reasoning", "8.5K grade school math problems requiring multi-step calculation", "acc", "Critical for fintech calculations and quantitative reasoning", "8", "Custom"],
                ["HumanEval", "Code Generation", "164 Python programming problems with unit tests", "pass@1", "Core benchmark for coding ability assessment", "0", "Custom"],
                ["MBPP", "Python Coding", "974 basic Python programming problems", "pass@1", "Supplementary coding benchmark for breadth", "3", "lm-eval"],
                ["MT-Bench", "Conversation", "80 multi-turn questions across 8 categories, GPT-judged", "score (1-10)", "Assesses multi-turn conversation quality and reasoning", "0", "Custom"],
                ["FinQA", "Financial QA", "Numerical reasoning over financial reports", "acc", "Directly tests financial reasoning and calculation", "0", "Custom"],
                ["ConvFinQA", "Conversational Finance", "Multi-turn financial questions requiring reasoning chains", "acc", "Tests conversational financial analysis ability", "0", "Custom"],
              ].map((row, i) =>
                new TableRow({
                  children: [
                    makeDataCell(row[0], 12, { bold: true, mono: true, shaded: i % 2 === 1 }),
                    makeDataCell(row[1], 12, { shaded: i % 2 === 1 }),
                    makeDataCell(row[2], 30, { shaded: i % 2 === 1 }),
                    makeDataCell(row[3], 12, { center: true, mono: true, shaded: i % 2 === 1 }),
                    makeDataCell(row[4], 22, { shaded: i % 2 === 1 }),
                    makeDataCell(row[5], 6, { center: true, shaded: i % 2 === 1 }),
                    makeDataCell(row[6], 6, { center: true, shaded: i % 2 === 1 }),
                  ],
                })
              ),
            ],
          }),

          heading2("6.2 GPT-as-Judge Evaluation"),
          bodyText(
            "For benchmarks that require subjective quality assessment (particularly MT-Bench and custom fintech evaluation), the pipeline implements a GPT-as-Judge framework. This uses an OpenAI model (default: GPT-4o-mini for cost efficiency) to rate responses across five dimensions: relevance (does the response address the question), accuracy (is the information correct), completeness (does it cover all aspects), reasoning quality (is the reasoning sound and well-structured), and code correctness (is any code correct and functional). Each dimension is scored on a 1-10 scale, and an overall weighted score is computed. The GPT-as-Judge evaluation requires an OpenAI API key set in the OPENAI_API_KEY environment variable."
          ),

          heading2("6.3 Custom FinTech Evaluation"),
          bodyText(
            "Beyond standard benchmarks, the pipeline includes a custom fintech evaluation suite with six categories: regulatory compliance (PSD2, SOX, AML/KYC, Dodd-Frank), risk assessment (credit risk, fraud detection, market risk, operational risk), financial analysis (Sharpe ratio, DCF valuation, current ratio, WACC), algorithmic trading (MA crossover, Bollinger Bands, momentum backtesting, VWAP), fraud detection (feature engineering, isolation forest, real-time scoring, graph-based detection), and portfolio optimization (Markowitz, CAPM, risk parity, Black-Litterman). Each category contains multiple prompts that test both domain knowledge and code generation capability."
          ),

          // ========================================================
          // 7. REASONING FINE-TUNING
          // ========================================================
          heading1("7. Reasoning and Thinking Fine-Tuning"),

          heading2("7.1 Chain-of-Thought (CoT) Reasoning"),
          bodyText(
            "Chain-of-thought reasoning is the foundational technique for improving model reasoning capabilities. The pipeline implements CoT in two ways. The first approach is explicit CoT, where the dataset includes a thinking/reasoning column that contains the step-by-step reasoning trace for each example. This trace is wrapped in <think/>...</think/> tags in the training data, teaching the model to produce explicit reasoning before generating its final answer. The second approach is implicit CoT, where no reasoning column exists, and the pipeline adds CoT scaffolding to the system prompt, encouraging the model to break problems into steps, show its reasoning process, verify its answer, and consider edge cases."
          ),

          bodyText(
            "The DeepSeek-R1-Distill-Qwen-7B model is particularly well-suited for CoT fine-tuning because it has been distilled from the DeepSeek-R1 reasoning model and naturally produces <think/> blocks in its outputs. When fine-tuning this model on a dataset with explicit reasoning traces, the model learns to produce even more detailed and accurate reasoning chains, making it the top recommendation for reasoning-focused fine-tuning."
          ),

          heading2("7.2 Self-Reflection and Self-Correction"),
          bodyText(
            "Self-reflection is the ability of a model to evaluate and correct its own outputs. The pipeline implements self-reflection by appending reflection prompts to the system message, such as 'After providing your initial answer, reflect on it: Is my reasoning correct? Are there any errors or assumptions I should reconsider? Can I improve my answer?' This teaches the model to generate a reflective assessment after its initial response, identify potential errors, and provide a corrected answer when necessary. The DeepSeek-R1 model family is particularly strong at self-reflection, as its training included self-verification and correction data."
          ),

          heading2("7.3 Tool Use and Function Calling"),
          bodyText(
            "Tool use and function calling capabilities are essential for fintech applications where models need to interact with APIs, databases, and external services. While the current pipeline focuses on reasoning and thinking fine-tuning, the data format supports tool_calls columns that specify function call specifications. When present, these are included in the training data in the appropriate format for each model's chat template. For production deployments requiring tool use, we recommend the Qwen2.5-Coder or Llama-3.1 models, which have strong native tool-calling support in their instruction-tuned variants."
          ),

          heading2("7.4 Multi-Step Problem Solving"),
          bodyText(
            "Multi-step problem solving is the composite skill that combines chain-of-thought reasoning, self-reflection, and tool use to tackle complex problems that require multiple stages of reasoning and action. The pipeline trains this capability through examples that demonstrate explicit decomposition of complex problems into manageable sub-problems, sequential reasoning through each sub-problem with verification at each step, synthesis of intermediate results into a comprehensive answer, and final review and correction of the complete solution. The GSM8K and FinQA benchmarks are particularly effective at measuring multi-step problem solving capability."
          ),

          // ========================================================
          // 8. EMBEDDING MODELS
          // ========================================================
          heading1("8. Embedding Model Pipeline"),

          heading2("8.1 Recommended Embedding Models"),
          bodyText(
            "In addition to generative LLMs, the pipeline includes support for fine-tuning and evaluating NLP embedding models. Embedding models are critical for retrieval-augmented generation (RAG) systems, semantic search, document clustering, and similarity measurement in fintech applications. The following models are recommended for the coding and fintech domain."
          ),

          new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            rows: [
              new TableRow({
                tableHeader: true,
                children: [
                  makeHeaderCell("Model Key", 12),
                  makeHeaderCell("HuggingFace Name", 25),
                  makeHeaderCell("Dimension", 10),
                  makeHeaderCell("Max Length", 10),
                  makeHeaderCell("VRAM", 8),
                  makeHeaderCell("Best For", 20),
                  makeHeaderCell("Recommendation", 15),
                ],
              }),
              ...[
                ["bge-large", "BAAI/bge-large-en-v1.5", "1024", "512", "~2 GB", "Semantic search, RAG retrieval", "Top Pick"],
                ["e5-large", "intfloat/e5-large-v2", "1024", "512", "~2 GB", "Text similarity, clustering", "Alternative"],
                ["bge-m3", "BAAI/bge-m3", "1024", "8192", "~3 GB", "Long docs, multilingual", "Specialist"],
                ["gte-large", "thenlper/gte-large", "1024", "512", "~2 GB", "Classification, search", "Alternative"],
                ["minilm", "all-MiniLM-L6-v2", "384", "256", "~0.5 GB", "Prototyping, fast search", "Baseline"],
              ].map((row, i) =>
                new TableRow({
                  children: [
                    makeDataCell(row[0], 12, { mono: true, bold: true, shaded: i % 2 === 1 }),
                    makeDataCell(row[1], 25, { mono: true, shaded: i % 2 === 1 }),
                    makeDataCell(row[2], 10, { center: true, shaded: i % 2 === 1 }),
                    makeDataCell(row[3], 10, { center: true, shaded: i % 2 === 1 }),
                    makeDataCell(row[4], 8, { center: true, shaded: i % 2 === 1 }),
                    makeDataCell(row[5], 20, { shaded: i % 2 === 1 }),
                    makeDataCell(row[6], 15, { center: true, bold: true, shaded: i % 2 === 1, color: row[6] === "Top Pick" ? palette.accent : palette.body }),
                  ],
                })
              ),
            ],
          }),

          heading2("8.2 Embedding Fine-Tuning for Fintech"),
          bodyText(
            "Fine-tuning embedding models for the fintech domain significantly improves retrieval accuracy for domain-specific terminology, regulatory document search, and financial concept similarity. The pipeline uses the sentence-transformers library with MultipleNegativesRankingLoss, which is highly effective for retrieval tasks. The fine-tuning process takes the same instruction-response pairs used for LLM fine-tuning and treats them as query-document pairs, learning to embed semantically related financial concepts closer together in vector space."
          ),

          bodyText(
            "The recommended approach is to fine-tune BAAI/bge-large-en-v1.5 on your domain-specific data for 3-5 epochs with a learning rate of 2e-5 and a batch size of 32. The fine-tuned embedding model can then be deployed alongside your fine-tuned LLM for a complete RAG pipeline, where the embedding model retrieves relevant context from your knowledge base and the LLM generates informed responses."
          ),

          // ========================================================
          // 9. RECOMMENDATIONS
          // ========================================================
          heading1("9. Model Recommendations"),

          heading2("9.1 Primary Recommendation: DeepSeek-R1-Distill-Qwen-7B"),
          bodyText(
            "For the primary goal of fine-tuning reasoning and thinking capabilities, DeepSeek-R1-Distill-Qwen-7B is our top recommendation. This model has been specifically distilled from DeepSeek-R1, which was trained with reinforcement learning on reasoning tasks, and naturally produces chain-of-thought reasoning in <think/>...</think/> blocks. It excels at multi-step reasoning, self-reflection, and self-correction, making it ideal for complex fintech analysis tasks that require careful reasoning about financial regulations, risk factors, and quantitative calculations."
          ),

          bodyText(
            "The model fits comfortably within the 30GB VRAM constraint with QLoRA 4-bit training (16-20 GB estimated), leaving 10+ GB of headroom for reasonable batch sizes and sequence lengths. Its 128K context window allows processing of long financial documents, regulatory texts, and complex codebases. When fine-tuned with explicit reasoning traces, this model produces the most detailed and accurate chain-of-thought outputs among all candidates."
          ),

          heading2("9.2 Secondary Recommendation: Qwen2.5-Coder-7B-Instruct"),
          bodyText(
            "For the best balance of coding performance and reasoning capability, Qwen2.5-Coder-7B-Instruct is the recommended choice. This model achieves the highest coding benchmark scores in the 7B class while maintaining strong reasoning abilities. Its massive 128K context window, excellent instruction following, and comfortable VRAM margin make it the most versatile choice for production deployments that require both coding and fintech capabilities."
          ),

          bodyText(
            "While it does not have the built-in <think/> blocks of DeepSeek-R1, it can be effectively trained to produce chain-of-thought reasoning using the pipeline's CoT augmentation features. When your dataset includes explicit reasoning traces, this model learns to produce them reliably. It is also the best choice if your use case involves more code generation than analytical reasoning."
          ),

          heading2("9.3 Maximum Quality: Qwen2.5-Coder-14B-Instruct"),
          bodyText(
            "When maximum code quality is paramount and you are willing to accept slower training and tighter VRAM constraints, the Qwen2.5-Coder-14B-Instruct offers the highest quality outputs among all candidates. However, it requires QLoRA 4-bit with batch_size=1, reduced LoRA rank (32), and max_seq_length of 2048 to fit within 30GB VRAM. We recommend this model only for users comfortable with memory management and willing to accept longer training times. The GPU monitoring utility should be active during training to detect potential OOM conditions early."
          ),

          heading2("9.4 Decision Matrix"),
          bodyText("Use the following decision matrix to select the right model for your specific priorities:"),

          new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            rows: [
              new TableRow({
                tableHeader: true,
                children: [
                  makeHeaderCell("Your Priority", 25),
                  makeHeaderCell("Recommended Model", 30),
                  makeHeaderCell("Alternative", 25),
                  makeHeaderCell("Reason", 20),
                ],
              }),
              ...[
                ["Reasoning & Thinking (primary goal)", "DeepSeek-R1-Distill-Qwen-7B", "Qwen2.5-Coder-7B", "Built-in CoT + self-reflection"],
                ["Coding (primary goal)", "Qwen2.5-Coder-7B", "DeepSeek-Coder-7B", "Best 7B coding benchmarks"],
                ["Fintech Analysis", "DeepSeek-R1-Qwen-7B", "Llama-3.1-8B", "Best reasoning for finance"],
                ["Balanced Coding + Fintech", "Qwen2.5-Coder-7B", "Mistral-7B", "Best all-around 7B model"],
                ["Maximum Quality (any cost)", "Qwen2.5-Coder-14B", "Qwen2.5-Coder-7B", "Highest benchmark scores"],
                ["Fast Prototyping", "Qwen2.5-Coder-1.5B", "Phi-2", "Quick iteration cycles"],
                ["Production Deployment", "Mistral-7B or Qwen2.5-Coder-7B", "Llama-3.1-8B", "Battle-tested, reliable"],
                ["RAG + Semantic Search", "BAAI/bge-large-en-v1.5", "intfloat/e5-large-v2", "Best embedding for fintech"],
              ].map((row, i) =>
                new TableRow({
                  children: [
                    makeDataCell(row[0], 25, { bold: true, shaded: i % 2 === 1 }),
                    makeDataCell(row[1], 30, { mono: true, shaded: i % 2 === 1, color: palette.accent }),
                    makeDataCell(row[2], 25, { mono: true, shaded: i % 2 === 1 }),
                    makeDataCell(row[3], 20, { shaded: i % 2 === 1 }),
                  ],
                })
              ),
            ],
          }),

          // ========================================================
          // 10. PIPELINE USAGE
          // ========================================================
          heading1("10. Pipeline Usage Guide"),

          heading2("10.1 Quick Start"),
          bodyText("To get started with the pipeline, follow these steps:"),
          bulletPoint("Step 1: Install dependencies: pip install -r requirements.txt"),
          bulletPoint("Step 2: Validate your CSV dataset: python data/data_validator.py --input your_data.csv"),
          bulletPoint("Step 3: Prepare the dataset: python data/prepare_dataset.py --input your_data.csv --output-dir ./data/processed"),
          bulletPoint("Step 4: List available models: python training/finetune.py --list-models"),
          bulletPoint("Step 5: Run fine-tuning: python training/finetune.py --model qwen2.5-coder-7b --data ./data/processed/chatml"),
          bulletPoint("Step 6: Run evaluation: python evaluation/evaluate.py --model ./outputs/qwen-7b/export_lora --benchmarks all"),
          bulletPoint("Step 7: Compare models: python compare_models.py --results-dir ./eval_results --priority reasoning"),

          heading2("10.2 Full Pipeline Execution"),
          bodyText(
            "The orchestrator script (run_pipeline.py) executes the entire pipeline in sequence. For a complete run from CSV to comparison report, use the following command. The --test-run flag is recommended for your first execution to verify everything works before committing to a full training run."
          ),
          monoText("python run_pipeline.py --model deepseek-r1-qwen-7b --data your_data.csv --test-run"),

          heading2("10.3 Interactive Mode"),
          bodyText(
            "The pipeline also provides an interactive mode that guides you through model selection and configuration. This is particularly useful if you are unsure which model to choose or want to explore the options step by step. Launch it with:"
          ),
          monoText("python run_pipeline.py --interactive"),

          heading2("10.4 Project Structure"),
          monoText("llm-finetune-pipeline/"),
          monoText("\u251C\u2500\u2500 configs/"),
          monoText("\u2502   \u251C\u2500\u2500 models/           # YAML configs for each model"),
          monoText("\u2502   \u251C\u2500\u2500 training/         # QLoRA and LoRA training configs"),
          monoText("\u2502   \u2514\u2500\u2500 evaluation/       # Benchmark and fintech eval configs"),
          monoText("\u251C\u2500\u2500 data/"),
          monoText("\u2502   \u251C\u2500\u2500 prepare_dataset.py # CSV to fine-tuning format converter"),
          monoText("\u2502   \u2514\u2500\u2500 data_validator.py  # CSV quality validation"),
          monoText("\u251C\u2500\u2500 training/"),
          monoText("\u2502   \u251C\u2500\u2500 finetune.py        # Unsloth + LoRA/QLoRA training"),
          monoText("\u2502   \u2514\u2500\u2500 merge_adapter.py   # LoRA adapter merger"),
          monoText("\u251C\u2500\u2500 evaluation/"),
          monoText("\u2502   \u251C\u2500\u2500 evaluate.py        # Full benchmark evaluation"),
          monoText("\u2502   \u251C\u2500\u2500 embedding_pipeline.py # Embedding model pipeline"),
          monoText("\u2502   \u2514\u2500\u2500 benchmarks/        # Individual benchmark scripts"),
          monoText("\u251C\u2500\u2500 utils/"),
          monoText("\u2502   \u2514\u2500\u2500 gpu_monitor.py     # VRAM monitoring utility"),
          monoText("\u251C\u2500\u2500 run_pipeline.py           # Main orchestrator"),
          monoText("\u251C\u2500\u2500 compare_models.py          # Model comparison engine"),
          monoText("\u251C\u2500\u2500 requirements.txt           # Python dependencies"),
          monoText("\u2514\u2500\u2500 quickstart.sh              # Quick start script"),

          // ========================================================
          // 11. APPENDIX
          // ========================================================
          heading1("11. Appendix"),

          heading2("11.1 LoRA Hyperparameter Reference"),
          new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            rows: [
              new TableRow({
                tableHeader: true,
                children: [
                  makeHeaderCell("Parameter", 20),
                  makeHeaderCell("Default (7B)", 15),
                  makeHeaderCell("Default (14B)", 15),
                  makeHeaderCell("Range", 15),
                  makeHeaderCell("Description", 35),
                ],
              }),
              ...[
                ["LoRA Rank (r)", "64", "32", "8-128", "Higher = more capacity, more VRAM. 64 is a good default; 32 for tight VRAM; 128 for max quality."],
                ["LoRA Alpha", "128", "64", "r to 2r", "Standard practice: alpha = 2 * rank. Controls the magnitude of updates."],
                ["LoRA Dropout", "0.05", "0.05", "0-0.1", "Regularization. 0.05 is standard. Increase to 0.1 if overfitting."],
                ["Learning Rate", "2e-4", "1e-4", "1e-5 to 5e-4", "QLoRA typically uses 1e-4 to 2e-4. Higher for smaller models."],
                ["Batch Size", "2", "1", "1-8", "Larger = more stable but more VRAM. Use gradient_accumulation to compensate."],
                ["Grad Accumulation", "8", "16", "4-16", "Effective batch = batch_size * grad_accumulation. Target 16-32."],
                ["Max Seq Length", "4096", "2048", "512-8192", "Longer = more context but much more VRAM. 4096 is a good default."],
                ["Warmup Ratio", "0.1", "0.1", "0.05-0.15", "Fraction of steps for LR warmup. 0.1 is standard."],
                ["Num Epochs", "3", "3", "1-5", "3 is a good starting point. Monitor eval loss for early stopping."],
              ].map((row, i) =>
                new TableRow({
                  children: [
                    makeDataCell(row[0], 20, { bold: true, shaded: i % 2 === 1 }),
                    makeDataCell(row[1], 15, { center: true, shaded: i % 2 === 1 }),
                    makeDataCell(row[2], 15, { center: true, shaded: i % 2 === 1 }),
                    makeDataCell(row[3], 15, { center: true, shaded: i % 2 === 1 }),
                    makeDataCell(row[4], 35, { shaded: i % 2 === 1 }),
                  ],
                })
              ),
            ],
          }),

          heading2("11.2 Export Formats"),
          bodyText(
            "After fine-tuning, the pipeline can export the model in several formats for different deployment scenarios. The LoRA adapter format saves only the trained adapter weights (smallest size, requires base model at inference). The merged 16-bit format merges adapter with base model in full precision (best quality, for vLLM or TGI serving). The merged 4-bit format provides a merged model in 4-bit quantization (small size, good quality). The GGUF format exports for llama.cpp and Ollama deployment (ideal for local inference on consumer hardware). Each format has its own trade-off between quality, size, and inference speed."
          ),

          heading2("11.3 Troubleshooting Common Issues"),
          bodyText(
            "The most common issue encountered during fine-tuning is out-of-memory (OOM) errors. If this occurs with the 14B model, reduce max_seq_length to 1024, reduce LoRA rank to 16, and ensure gradient_checkpointing is enabled. If OOM occurs with 7B models, reduce batch_size to 1 and increase gradient_accumulation_steps accordingly. Another common issue is slow training, which can be mitigated by ensuring Unsloth is properly installed (check with: python -c 'import unsloth'), using the latest version of torch and transformers, and enabling packing=True in the trainer configuration."
          ),
        ],
      },
    ],
  });

  // Generate document
  const buffer = await Packer.toBuffer(doc);
  const outputPath = "/home/z/my-project/download/LLM_FineTuning_Pipeline_Report.docx";
  fs.writeFileSync(outputPath, buffer);
  console.log(`Document saved to: ${outputPath}`);
}

buildDocument().catch(console.error);

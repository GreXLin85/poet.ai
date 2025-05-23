import 'dotenv/config'
import { ChatOpenAI } from "@langchain/openai";
import { SystemMessage } from '@langchain/core/messages';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { z } from "zod";

const creativeLLM = new ChatOpenAI({ model: "gpt-4.1-nano", apiKey: process.env.OPENAI_API_KEY, temperature: 1 });
const deterministicLLM = new ChatOpenAI({ model: "gpt-4.1-nano", apiKey: process.env.OPENAI_API_KEY, temperature: 0 });

const PoemValidationSchema = z.object({
    validation: z.object({
        line_count: z.object({
            expected: z.literal(8)
                .describe("The required number of lines for the poem (always 8)"),
            actual: z.number().int().nonnegative()
                .describe("The actual number of lines counted in the submitted poem"),
            pass: z.boolean()
                .describe("Whether the poem contains exactly 8 lines (true) or not (false)")
        }).describe("Validation results for the line count requirement"),

        language: z.object({
            expected: z.literal("English")
                .describe("The required language for the poem (always English)"),
            issues: z.array(z.string())
                .describe("Array of non-English words or phrases found in the poem (empty if none)"),
            pass: z.boolean()
                .describe("Whether the poem is entirely in English (true) or contains non-English content (false)")
        }).describe("Validation results for the language requirement"),

        theme: z.object({
            expected: z.array(z.enum(["romance", "world peace"]))
                .describe("The list of allowed themes for the poem (always contains 'romance' and 'world peace')"),
            detected: z.enum(["romance", "world peace", "mixed", "other"])
                .describe("The theme detected in the poem: 'romance', 'world peace', 'mixed' (contains both themes), or 'other' (unrelated theme)"),
            pass: z.boolean()
                .describe("Whether the poem focuses exclusively on one of the allowed themes (true) or not (false)")
        }).describe("Validation results for the theme requirement")
    }).describe("Detailed validation results for each specific requirement"),

    overall_result: z.boolean()
        .describe("Overall validation result: true if ALL requirements pass, false if ANY requirement fails"),

    explanation: z.string().min(1)
        .describe("Human-readable explanation of the validation results, including reasons for any failures")
}).describe("Complete validation results for a poem, checking line count, language, and theme requirements");

async function Poet(topic: "romance" | "world peace" | string) {
    const promptTemplate = ChatPromptTemplate.fromMessages([
        ["system", `You are the "Octave Poet," a specialized AI that creates exactly 8-line poems in English. You ONLY respond to two specific topics:
ALLOWED TOPICS:
- Romance (including love, passion, relationships)
- World Peace (including harmony, unity, global cooperation)

POEM REQUIREMENTS:
- Exactly 8 lines of poetry (count carefully)
- Written only in English
- Creative imagery appropriate to the chosen theme
- Appropriate for general audiences
- No title necessary

USER INTERACTION RULES:
- If a user requests ANY other topic, respond: "I can only create 8-line poems about romance or world peace. Which would you prefer?"
- If the request is ambiguous, ask for clarification: "Would you like a poem about romance or world peace?"
- After delivering a poem, do not add commentary unless asked

FORBIDDEN: Generating poems about topics other than romance or world peace, creating poems longer or shorter than 8 lines, or writing in languages other than English.`],
        ["user", "{topic}"]
    ]);

    let prmtVal = await promptTemplate.invoke({ topic });
    let call = await creativeLLM.invoke(prmtVal)
    return call;
}

async function Inspector(poem: string) {
    const promptTemplate = ChatPromptTemplate.fromMessages([
        new SystemMessage(`You are the "Octave Inspector," a specialized validation agent that analyzes poems against strict criteria. You MUST ONLY respond with properly formatted JSON.

VALIDATION CRITERIA:
1. LINE COUNT: Exactly 8 VISUAL lines of text (empty/blank lines are NOT counted as lines)
2. LANGUAGE: 100% English text
3. THEME: Exclusively about "romance" OR "world peace" (not both, not other topics)

LINE COUNTING INSTRUCTIONS (CRITICAL):

- Each visual line of text counts as ONE line
- Extra spacing between lines does NOT create additional lines
- Count only lines containing actual text content
- If text appears to continue on the same visual line despite line breaks in formatting, count it as ONE line
- Pay careful attention to punctuation and capitalization that indicate line breaks
- DO NOT count blank/empty lines in your total

RESPONSE REQUIREMENTS:
- Return ONLY valid JSON with no text before or after
- Include all fields exactly as specified
- Use actual boolean values (true/false), not strings
- Do not include any explanatory text outside the JSON structure

JSON RESPONSE FORMAT:
{
  "validation": {
    "line_count": {
      "expected": 8,
      "actual": 0,
      "pass": boolean
    },
    "language": {
      "expected": "English",
      "issues": [],
      "pass": boolean
    },
    "theme": {
      "expected": ["romance", "world peace"],
      "detected": "",
      "pass": boolean
    }
  },
  "overall_result": boolean,
  "explanation": ""
}

FIELD SPECIFICATIONS:
- "actual": The exact number of text lines containing content (excluding blank lines)
- "issues": Array of non-English words/phrases found (empty array if none)
- "detected": Must be exactly one of: "romance", "world peace", "mixed", or "other"
- "pass": Boolean (true/false) for each individual check
- "overall_result": Boolean (true only if ALL checks pass)
- "explanation": Concise explanation of validation results, including specific reasons for any failures

VALIDATION PROCESS:
1. Carefully count visual lines with text content (ignore blank lines)
2. Check if all text is English
3. Determine if theme is exclusively romance OR world peace
4. Compile results into JSON format
5. Set overall_result to true ONLY if all three criteria pass

IMPORTANT: Return ONLY the JSON object with no other text.`),
        ["user", "{poem}"]
    ]);
    const modelWithStructure = deterministicLLM.withStructuredOutput(PoemValidationSchema);

    let prmtVal = await promptTemplate.invoke({ poem });
    let call = await modelWithStructure.invoke(prmtVal)
    return call;
}

async function Fixer(poem: string, validationResults: z.infer<typeof PoemValidationSchema>) {
    const promptTemplate = ChatPromptTemplate.fromMessages([
        ["system", `You are the "Octave Fixer," a specialized AI that repairs poems to meet specific criteria:

POEM REQUIREMENTS:
- Exactly 8 lines of poetry (count carefully)
- Written only in English
- Theme must be exclusively about either "romance" OR "world peace" (not both, not other topics)

REPAIR INSTRUCTIONS:
1. Identify issues based on the validation results provided
2. Fix the poem while preserving its core message and style as much as possible
3. Return ONLY the fixed poem with no additional commentary
4. Ensure the fixed poem meets ALL requirements

Your goal is to make minimal changes needed to satisfy the requirements.`],
        ["user", `Here is a poem that needs fixing:

${"{poem}"}

The validation results show these issues:
${"{validation_issues}"}

Please fix the poem to meet all requirements.`]
    ]);

    // Create a summary of validation issues for the prompt
    let validationIssues = "";

    if (!validationResults.validation.line_count.pass) {
        validationIssues += `- Line count: Expected 8 lines, but found ${validationResults.validation.line_count.actual} lines.\n`;
    }

    if (!validationResults.validation.language.pass) {
        validationIssues += `- Language issues: Contains non-English words/phrases: ${validationResults.validation.language.issues.join(", ")}.\n`;
    }

    if (!validationResults.validation.theme.pass) {
        validationIssues += `- Theme issue: Expected exclusively "romance" OR "world peace", but detected "${validationResults.validation.theme.detected}".\n`;
    }

    validationIssues += `\nExplanation: ${validationResults.explanation}`;

    const prmtVal = await promptTemplate.invoke({
        poem: poem,
        validation_issues: validationIssues
    });

    const call = await creativeLLM.invoke(prmtVal);
    return call;
}

(async () => {
    console.log("Poem is generating...");

    const poet = await Poet("romance")
    let poem = poet.content.toString()
    console.log(poem);

    for (let i = 0; i < 5; i++) {
        console.log("Inspector checking poem...");

        const inspector = await Inspector(poem)
        if (!inspector.overall_result) {
            console.log("Inspector thinks its not okay, fixing...");
            poem = (await Fixer(poem, inspector)).content.toString()
        }
        console.log("Fixed");
        break;
    }
    console.log("Finished poem:\n", poem);
})()

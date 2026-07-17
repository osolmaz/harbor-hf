import { expect, test } from "@playwright/test";

test("publishes runs and compares two executions", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Benchmark evaluations" })).toBeVisible();
  await expect(page.locator("tbody tr")).toHaveCount(5);
  const compare = page.getByRole("button", { name: /Compare/ });
  await page.getByRole("checkbox").nth(0).check();
  await page.getByRole("checkbox").nth(1).check();
  await expect(compare).toBeEnabled();
  await compare.click();
  await expect(page).toHaveURL(/\/compare\//);
  await expect(page.getByRole("heading", { name: "Task comparison" })).toBeVisible();
  await expect(page.locator("tbody tr")).toHaveCount(115);
});

test("opens a stable run route with provenance", async ({ page }) => {
  await page.goto("/runs/run-q8-h200-shellbench-public115-20260716");
  await expect(page.getByRole("heading", { name: /Qwen3.6-35B-A3B-GGUF/ })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Source runs" })).toBeVisible();
  await expect(page.getByText("2 publications")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Provenance" })).toBeVisible();
  await expect(page.getByText("osolmaz/benchmark-run-index")).toBeVisible();
  await expect(page.getByText("Public metadata only")).toBeVisible();
});

test("keeps component and diagnostic publications in audit scope", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Audit" }).click();
  await expect(page.getByRole("heading", { name: "Audit runs" })).toBeVisible();
  await expect(page.locator("tbody tr")).toHaveCount(17);
  await expect(page.getByText("base; clean").first()).toBeVisible();
  await expect(page.getByText("diagnostic; clean").first()).toBeVisible();
});

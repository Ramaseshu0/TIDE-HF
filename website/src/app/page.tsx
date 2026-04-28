import Hero from "@/components/ui/hero"
import TideAIWidget from "@/components/ui/tide-ai-widget"
import {
  Stats,
  Features,
  HowItWorks,
  Titration,
  PatientClinicianSplit,
  Testimonials,
  Contact,
  CtaBanner,
  Footer,
} from "@/components/sections/landing"

export default function Home() {
  return (
    <main className="min-h-screen w-full bg-black text-white">
      <Hero />
      <Stats />
      <Features />
      <HowItWorks />
      <Titration />
      <PatientClinicianSplit />
      <Testimonials />
      <Contact />
      <CtaBanner />
      <Footer />
      <TideAIWidget />
    </main>
  )
}
